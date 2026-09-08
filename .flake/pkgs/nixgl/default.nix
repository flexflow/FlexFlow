{ applyPatches
, pkgs
, src
}:

# Need a not-yet-merged PR for nixGL compatibility with NixOS 26.05.
#
# This evaluates to nixGL's package set, not a derivation, so it belongs in a
# let binding rather than in the flake's `packages` output.
import
  (applyPatches {
    name = "nixGL-patched-source";
    inherit src;
    patches = [
      ./drop-kernel-override.patch
      ./detect-open-kernel-module-version.patch
    ];
  })
  {
    inherit pkgs;
  }
