# Bazel Template

## Getting Started

### Install Bazelisk
Install Bazelisk by following the instructions on the [Bazelisk GitHub page](https://github.com/bazelbuild/bazelisk).

#### Bazelisk on macOS
```bash
brew install bazelisk
```

#### Bazelisk on Ubuntu
If you do not have curl, install it using:
```bash
sudo apt-get install curl
```
Then download Bazelisk (choose the correct architecture):
##### For x86_64 systems:
```bash
curl -LO "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64"
```
##### For ARM64 systems (e.g., Apple Silicon or newer ARM-based Linux machines):
```bash
curl -LO "https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-arm64"
```

Make bazelisk executable
```bash
chmod +x bazelisk-linux-amd64
```
Move Bazelisk to a directory in your PATH:
```bash
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
```
Verify the installation (this will install the appropriate Bazel version and print it):
```bash
bazel --version
```

### To Run Examples
From the root directory of the project, run:

```bash
bazel run //path/to/package:target_name
```

Where:
- `path/to/package` is the relative path to the directory containing the BUILD or BUILD.bazel file.
- `target_name` is the name attribute of a rule like py_binary, cc_binary, or sh_binary.