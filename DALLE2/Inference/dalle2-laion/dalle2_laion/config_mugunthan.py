from pathlib import Path
from dalle2_pytorch.train_configs import AdapterConfig as ClipConfig
from typing import List, Optional, Union
from enum import Enum
# from pydantic import BaseModel, model_validator, ValidationError
from pydantic import BaseModel, root_validator, ValidationError
from contextlib import contextmanager
import tempfile
import urllib.request
import json

class LoadLocation(str, Enum):
    local = "local"
    url = "url"

class File(BaseModel):
    load_type: LoadLocation
    path: str
    checksum_file_path: Optional[str] = None
    cache_dir: Optional[Path] = None
    filename_override: Optional[str] = None

    @root_validator(pre=True)
    # @model_validator(mode="after")
    # def add_default_checksum(cls, values):
    def set_default_checksum_path(self):
        if self.load_type == LoadLocation.url:
            if self.path.startswith("https://huggingface.co/") and "resolve" in self.path and self.checksum_file_path is None:
                self.checksum_file_path = self.path.replace("resolve/main/", "raw/main/")
        return self

    def download_to(self, path: Path):
        assert self.load_type == LoadLocation.url
        urllib.request.urlretrieve(self.path, path)
        if self.checksum_file_path is not None:
            urllib.request.urlretrieve(self.checksum_file_path, str(path) + ".checksum")

    def download_checksum_to(self, path: Path):
        assert self.load_type == LoadLocation.url
        assert self.checksum_file_path is not None, "No checksum file path specified"
        urllib.request.urlretrieve(self.checksum_file_path, path)

    def get_remote_checksum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.download_checksum_to(tmpdir + "/checksum")
            with open(tmpdir + "/checksum", "r") as f:
                checksum = f.read()
        return checksum

    @property
    def filename(self):
        if self.filename_override is not None:
            return self.filename_override
        filename = self.path.split('/')[-1]
        if '?' in filename:
            filename = filename.split('?')[0]
        return filename

    @contextmanager
    def as_local_file(self, check_update: bool = True):
        if self.load_type == LoadLocation.local:
            yield self.path
        elif self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            file_path = self.cache_dir / self.filename
            cached_checksum_path = self.cache_dir / (self.filename + ".checksum")
            if not file_path.exists():
                print(f"Downloading {self.path} to {file_path}")
                self.download_to(file_path)
            else:
                if self.checksum_file_path is None:
                    print(f'{file_path} already exists. Skipping download. No checksum found so if you think this file should be re-downloaded, delete it and try again.')
                elif not cached_checksum_path.exists():
                    if check_update:
                        print(f"Checksum not found for {file_path}. Downloading it again.")
                        self.download_to(file_path)
                    else:
                        print(f"Checksum not found for {file_path}, but updates are disabled. Skipping download.")
                else:
                    new_checksum = self.get_remote_checksum()
                    with open(cached_checksum_path, "r") as f:
                        old_checksum = f.read()
                    if new_checksum != old_checksum:
                        if check_update:
                            print(f"Checksum mismatch. Deleting {file_path} and downloading again.")
                            file_path.unlink()
                            self.download_to(file_path)
                        else:
                            print(f"Checksums mismatched, but updates are disabled. Skipping download.")
            yield file_path
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile = tmpdir + "/" + self.filename
                self.download_to(tmpfile)
                yield tmpfile

class SingleDecoderLoadConfig(BaseModel):
    unet_numbers: List[int]
    default_sample_timesteps: Optional[List[int]] = None
    default_cond_scale: Optional[List[float]] = None
    load_model_from: File
    load_config_from: Optional[File]

class DecoderLoadConfig(BaseModel):
    unet_sources: List[SingleDecoderLoadConfig]
    final_unet_number: int

    @root_validator(pre=True)
    # @model_validator(mode="after")
    # def compute_num_unets(cls, values):
    def compute_final_unet_number(self):
        unet_numbers = []
        for value in self.unet_sources:
            unet_numbers.extend(value.unet_numbers)
        self.final_unet_number = max(unet_numbers)
        return self

    @root_validator
    # @model_validator(mode="after")
    # def verify_unet_numbers_valid(cls, values):
    def validate_unet_number_sequence(self):
        unet_numbers = []
        for value in self.unet_sources:
            unet_numbers.extend(value.unet_numbers)
        unet_numbers.sort()
        if len(unet_numbers) != len(set(unet_numbers)):
            raise ValidationError("The decoder unet numbers must not repeat.")
        if unet_numbers[0] != 1:
            raise ValidationError("The decoder unet numbers must start from 1.")
        differences = [unet_numbers[i] - unet_numbers[i - 1] for i in range(1, len(unet_numbers))]
        if any(diff != 1 for diff in differences):
            raise ValidationError("The decoder unet numbers must not skip any.")
        return self

class PriorLoadConfig(BaseModel):
    default_sample_timesteps: Optional[int] = None
    default_cond_scale: Optional[float] = None
    load_model_from: File
    load_config_from: Optional[File]

class ModelLoadConfig(BaseModel):
    decoder: Optional[DecoderLoadConfig] = None
    prior: Optional[PriorLoadConfig] = None
    clip: Optional[ClipConfig] = None

    devices: Union[List[str], str] = 'cuda:0'
    load_on_cpu: bool = True
    strict_loading: bool = True

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)
