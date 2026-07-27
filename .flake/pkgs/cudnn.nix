{ cudaPackages
, fetchurl
}:

# Pin the latest cuDNN version with support for NVIDIA Pascal GPUs.
cudaPackages.cudnn.overrideAttrs (finalAttrs: _: {
  version = "9.10.2.21";

  src = fetchurl {
    url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${finalAttrs.version}_cuda12-archive.tar.xz";
    hash = "sha256-0N78vExtrXEf9Mtm0lQDajAMkHGwfHtkGZqsq1NDE8E=";
  };
})
