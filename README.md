# Halide toolkits (Play with Halide)

For more detail about what Halide is, see http://halide-lang.org.

For API documentation see http://halide-lang.org/docs

## Build & Install

To build this project, see the notes below.

Build Status
============

| Linux                        |
|------------------------------|
| [![linux build status][1]][2]|

[1]: https://travis-ci.org/xbwee1024/play-with-halide.svg?branch=master
[2]: https://travis-ci.org/xbwee1024/play-with-halide

### Prerequisites
**Ninja** installed

For Unix-like systems,
**gcc** or **clang** installed

For windows,
**Visual Studio** installed

For Android,
**Android NDK** installed. **Android SDK** is not required now (build Android examples is disabled)

### Build Halide
For Unix-like systems, 
```bash
./script/build-halide.sh
# or add android to cross-compile Android
./script/build-halide.sh android
```
this will build the **Halide** library and install into /usr/local/

For windows,
```bash
.\script\build-halide.bat
# or add android to cross-compile Android
.\script\build-halide.bat android
```

### Build Project

The build procedure is simple, just run `build.sh` or `build.bat` for Windows:

When building finished, binaries will be stay in `out/install/bin`
