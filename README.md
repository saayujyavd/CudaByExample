# CUDA by Example - Code Repository

This is my collection of code examples from "CUDA by Example" by Sanders and Kandrot. If you're learning CUDA or want to reference the examples from the book, everything's organized here by chapter.

## What's in Here

- **Chp03**: Getting started - device properties, hello world
- **Chp04**: Basic parallel computation - julia sets, vector sums
- **Chp05**: Working with memory - bitmaps, dot products, ripple effects
- **Chp06**: Ray tracing with constant memory
- **Chp07**: Texture memory and heat transfer
- **Chp08**: GPU-CPU interaction and graphics stuff
- **Chp09**: Histogram computation (both CPU and GPU versions)
- **Chp10**: Memory management and CUDA streams
- **Chp11**: Pageable & Pagelocked Memory
- **Appendix**: Advanced Atomics & CPU Hash Table

## What You Need

- **NVIDIA CUDA Toolkit** (5.0 or newer - grab it from https://developer.nvidia.com/cuda-downloads)
- A **C/C++ compiler**:
  - Windows: Visual Studio
  - Linux: GCC/Clang
  - macOS: Apple Clang (limited support on newer versions)
- An **NVIDIA GPU** with up-to-date drivers
- (Optional) **OpenGL/GLUT** if you want to run the graphics examples

## Folder Layout

```
.
├── Chp03/ ... Chp11/    Chapter code (organized by chapter number)
├── Appendix/            A few extra examples
└── Dependencies/        Headers and libraries you might need
    ├── bin/             DLLs for graphics (glut32.dll, glut64.dll)
    ├── lib/             Libraries (.lib files)
    └── common/          Header files (book.h, gpu_anim.h, etc)
        └── GL/          OpenGL headers
```

### Windows Setup

#### Option A: System-Level Installation (Recommended)

**For 32-bit CUDA projects:**
```batch
# Copy DLL files to Windows System32 folder
copy Dependencies\bin\glut32.dll C:\Windows\System32\

# Copy lib files to Visual Studio installation (adjust path as needed)
copy Dependencies\lib\glut32.lib "C:\Program Files\Microsoft Visual Studio XX\VC\lib\"

# Copy header files to Visual Studio installation
copy Dependencies\common\*.h "C:\Program Files\Microsoft Visual Studio XX\VC\include\"
copy Dependencies\common\GL\*.h "C:\Program Files\Microsoft Visual Studio XX\VC\include\GL\"
```

**For 64-bit CUDA projects:**
```batch
# Copy DLL files to Windows SysWOW64 folder (for 64-bit compatibility)
copy Dependencies\bin\glut64.dll C:\Windows\SysWOW64\

# Copy lib files to Visual Studio installation (64-bit path)
copy Dependencies\lib\glut64.lib "C:\Program Files\Microsoft Visual Studio XX\VC\lib\x64\"

# Copy header files
copy Dependencies\common\*.h "C:\Program Files\Microsoft Visual Studio XX\VC\include\"
copy Dependencies\common\GL\*.h "C:\Program Files\Microsoft Visual Studio XX\VC\include\GL\"
```

**Note:** Replace `XX` with your Visual Studio version number (e.g., `15` for VS 2017, `16` for VS 2019, `17` for VS 2022).

#### Option B: Project-Level Setup (Alternative)

No system-level installation needed. Instead, configure your project to use local paths:

1. In Visual Studio, right-click your project → Properties
2. Navigate to **VC++ Directories**
3. Add to **Include Directories:**
   ```
   $(ProjectDir)\Dependencies\common
   ```
4. Add to **Library Directories:**
   ```
   $(ProjectDir)\Dependencies\lib
   ```
5. In **Linker → Input → Additional Dependencies**, add:
   ```
   glut32.lib
   ```
   (or `glut64.lib` for 64-bit)

6. Copy the appropriate DLL to your executable's directory:
   - For 32-bit: Copy `Dependencies\bin\glut32.dll` to your build output folder
   - For 64-bit: Copy `Dependencies\bin\glut64.dll` to your build output folder

### Linux Setup

#### Option A: System-Level Installation (Recommended)

**Ubuntu/Debian:**
```bash
# Install development packages
sudo apt-get install build-essential cuda-toolkit

# Copy dependencies to system include directory
sudo cp -r Dependencies/common/* /usr/local/include/

# Optional: Install system GLUT package (alternative to provided libraries)
sudo apt-get install freeglut3-dev
```

**Fedora/RHEL:**
```bash
# Install development packages
sudo dnf install gcc gcc-c++ cuda-toolkit

# Copy dependencies
sudo cp -r Dependencies/common/* /usr/local/include/

# Optional: Install system GLUT package
sudo dnf install freeglut-devel
```

#### Option B: Project-Level Setup (Alternative)

No system installation needed. Compile with local include paths:

```bash
# Basic compilation with project-level dependencies
nvcc -I./Dependencies/common -o output_name source_file.cu

# With graphics support (using provided libraries)
nvcc -I./Dependencies/common -L./Dependencies/lib -o output_name source_file.cu -lglut -lGL -lX11

# Or use system-installed GLUT without copying
nvcc -I./Dependencies/common -o output_name source_file.cu -lglut -lGL -lX11
```

### macOS Setup

#### Option A: System-Level Installation (Recommended)

```bash
# Install CUDA Toolkit (if available for your macOS version)
# Download from NVIDIA website: https://developer.nvidia.com/cuda-downloads

# Copy dependencies to system include directory
sudo cp -r Dependencies/common/* /usr/local/include/

# Optional: Install system OpenGL/GLUT support via Homebrew
brew install freeglut
```

#### Option B: Project-Level Setup (Alternative)

No system installation needed:

```bash
# Compile with local include paths
nvcc -I./Dependencies/common -o output_name source_file.cu

# With graphics support
nvcc -I./Dependencies/common -L./Dependencies/lib -o output_name source_file.cu -lglut -lGL

# Or use Homebrew-installed GLUT without copying
nvcc -I./Dependencies/common -o output_name source_file.cu -lglut -framework OpenGL
```

## 🔨 Building Examples

### Windows (Visual Studio)

#### Using System-Level Dependencies

1. Ensure you've completed the Windows system-level setup (Option A)
2. Create a CUDA project in Visual Studio
3. Link graphics library in **Linker → Input → Additional Dependencies:**
   - Add `glut32.lib` (32-bit) or `glut64.lib` (64-bit)
4. Build: **Build → Build Solution (F7)**

#### Using Project-Level Dependencies

1. Create a CUDA project in Visual Studio
2. Right-click Project → Properties → VC++ Directories
3. Add to **Include Directories:**
   ```
   $(ProjectDir)\Dependencies\common
   ```
4. Add to **Library Directories:**
   ```
   $(ProjectDir)\Dependencies\lib
   ```
5. Add to **Linker → Input → Additional Dependencies:**
   ```
   glut32.lib
   ```
   (or `glut64.lib` for 64-bit)
6. Copy DLL to output folder (see Windows setup Option B)
7. Build: **Build → Build Solution (F7)**

### Linux/macOS

#### Using System-Level Dependencies

```bash
# After system-level setup (Option A), compile directly
nvcc -o output_name source_file.cu

# With graphics
nvcc -o output_name source_file.cu -lglut -lGL -lX11
```

#### Using Project-Level Dependencies

```bash
# Specify paths at compile time (no setup required)
nvcc -I./Dependencies/common -o output_name source_file.cu

# With graphics support
nvcc -I./Dependencies/common -L./Dependencies/lib -o output_name source_file.cu -lglut -lGL -lX11

# Optimized build
nvcc -O3 -I./Dependencies/common -o output_name source_file.cu

# With debugging
nvcc -g -G -I./Dependencies/common -o output_name source_file.cu
```

## 🎯 Running Examples

### Simple Examples (No Graphics Required)

These compile and run without graphics dependencies:

```bash
# Basic examples
./Chp03/HelloWorld/hello
./Chp04/VecSum/OnGPU/vecsum
./Chp09/Histogram/OnGPU/histogram
# etc.
```

All examples in the following work without graphics libraries:
- Chp03 (Device Properties, Hello World)
- Chp04 (Julia set computation, Vector sum)
- Chp06 (Ray tracing)
- Chp07 (Heat transfer)
- Chp09 (Histogram)
- Chp10 (Memory and streams)
- Chp11 (Advanced techniques)
- Appendix (Additional examples)

### Graphics Examples (Optional - Requires OpenGL/GLUT)

The following examples can display visual output but are **optional**:
- `Chp05/Ripple` - CPU ripple animation
- `Chp05/Bitmap/Synchronized` - GPU bitmap computation
- `Chp08/GPURipple` - GPU ripple with graphics
- `Chp08/HypnoticPattern` - GPU pattern generation

**To compile graphics examples:**

```bash
# Linux/macOS with system-level setup
nvcc -o ripple ripple.cu -lglut -lGL -lX11

# Linux/macOS with project-level setup
nvcc -I../Dependencies/common -L../Dependencies/lib -o ripple ripple.cu -lglut -lGL -lX11
```

These will open a graphics window displaying the results. Close the window or press ESC to exit.

## ⚙️ Compiler Flags Reference

| Flag | Purpose |
|------|---------|
| `-O3` | Optimization level 3 (performance) |
| `-g` | Include debugging symbols |
| `-G` | Include GPU debugging symbols |
| `-lglut` | Link OpenGL/GLUT library |
| `-lGL` | Link OpenGL library |
| `-lX11` | Link X11 (display) library |
| `--gpu-architecture=compute_XX` | Target specific GPU architecture |

## 🐛 Troubleshooting

### Quick Diagnostic: Which Setup Approach Are You Using?

Check what approach you're following:

- **System-Level Installation**: Dependencies are copied to system directories
  - Windows: Check `C:\Windows\System32` or `C:\Program Files\...\include`
  - Linux: Check `/usr/local/include`
  - macOS: Check `/usr/local/include`

- **Project-Level Setup**: Using `-I./Dependencies/common` flags at compile time
  - Check: Are you passing `-I` include flags?
  - Check: Is your working directory in the repository root?

---

### Common Issues

**Issue: "nvcc not found" or "CUDA toolkit not installed"**
- Verify CUDA installation:
  - Windows: Run `nvcc --version` in Command Prompt
  - Linux/macOS: Run `which nvcc` or `nvcc --version`
- Add CUDA to PATH:
  - Windows: Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin` to system PATH
  - Linux: Add to `~/.bashrc`: `export PATH=/usr/local/cuda/bin:$PATH`
  - macOS: Add to `~/.bash_profile`: `export PATH=/usr/local/cuda/bin:$PATH`

**Issue: "book.h: No such file or directory" or "glut.h: No such file or directory"**

If using **System-Level Setup (Option A)**:
- Verify dependencies were copied correctly:
  - Linux/macOS: `ls /usr/local/include/book.h`
  - Windows: Check `C:\Program Files\...\include\book.h`

If using **Project-Level Setup (Option B)**:
- Ensure you're in the repository root directory
- Add `-I./Dependencies/common` flag:
  ```bash
  nvcc -I./Dependencies/common -o program source.cu
  ```
- Verify file exists: `ls Dependencies/common/book.h`

**Issue: Linker error for glut32.lib/glut64.lib**

If using **System-Level Setup (Option A)**:
- Verify library files are in system directories:
  - Windows: Check `C:\Program Files\...\lib\glut32.lib`
  - Linux: Libraries may be system-provided (not needed from Dependencies)

If using **Project-Level Setup (Option B)**:
- Add library paths to compile command:
  ```bash
  nvcc -L./Dependencies/lib -o program source.cu -lglut
  ```
- Windows Visual Studio: Add paths in **Project Properties → VC++ Directories → Library Directories**
- Verify files exist: `ls Dependencies/lib/glut*.lib`

**Issue: DLL not found (Windows runtime error like "glut32.dll not found")**

If using **System-Level Setup (Option A)**:
- DLL should be in `C:\Windows\System32` or `C:\Windows\SysWOW64`
- Verify: `dir C:\Windows\System32\glut32.dll`

If using **Project-Level Setup (Option B)**:
- DLL must be in the same folder as your executable
- Copy: `copy Dependencies\bin\glut32.dll <build_output_folder>`
- Or: Add to PATH environment variable

**Issue: GPU not detected**
- Run: `nvidia-smi` (Windows/Linux) to check GPU driver
- Verify CUDA-capable GPU is installed
- Update GPU driver to latest version
- Check CUDA compatibility: https://docs.nvidia.com/cuda/cuda-gdb/index.html#supported-platforms

**Issue: "undefined reference to '__cudaRegisterFatBinary'" or similar CUDA errors**
- Ensure you're compiling with `nvcc`, not just `gcc` or `cl`
- Windows: Use CUDA Visual Studio integration, not regular compiler
- Linux/macOS: Verify `nvcc` is being used: `which nvcc`

### Platform-Specific Notes

**Windows:**
- Visual Studio projects must be configured to use CUDA build tools
- CUDA 5.0+ required; recommend CUDA 10.0 or later
- 32-bit and 64-bit compilation require different library versions
- If using **Project-Level Setup**, both DLL and LIB must match (32/64-bit)

**Linux:**
- May require `sudo` for system-wide library installation (System-Level Option A)
- Use `ldd` to verify library dependencies: `ldd ./executable`
- Some distributions use different paths for CUDA installation
- Verify library location: `ldconfig -p | grep cuda`

**macOS:**
- CUDA support is limited on newer macOS versions (check NVIDIA compatibility)
- Homebrew can help manage graphics dependencies
- Path handling may differ; use `$(which nvcc)` to verify installation
- System-Level Setup may require additional flags for newer macOS versions

## Example Walkthroughs

### Running Chp03 - Hello World

```bash
cd Chp03/HelloWorld
nvcc -o hello HelloWorld.cu
./hello
# Output: 2 + 7 = 9
```

### Running Chp05 - Ripple with Graphics

```bash
cd Chp05/Ripple
nvcc -I../../Dependencies/common -o ripple ripple.cu -lglut -lGL -lX11
./ripple
# A graphics window will open displaying a ripple animation
```

### Running Chp09 - Histogram

```bash
cd Chp09/Histogram/OnGPU/GlobalMemAtomics
nvcc -o histogram main.cu
./histogram
# Outputs histogram computation results
```

## Learning Path

Recommended order for learning:

1. **Chp03**: Understanding CUDA basics and device properties
2. **Chp04**: Basic kernel launches and data transfers
3. **Chp05**: Memory management and synchronization
4. **Chp06-07**: Advanced memory types (constant, texture)
5. **Chp08**: GPU-CPU interaction and graphics
6. **Chp09**: Atomic operations and synchronization patterns
7. **Chp10**: Streams and advanced memory management
8. **Chp11**: Pageable & Pagelocked Memory

## External Resources

- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [GLUT Documentation](https://www.opengl.org/resources/libraries/glut/)
- [Original Book Website](https://developer.nvidia.com/cuda-example) (if available)

## Notes for Contributors

If you're adding new examples or modifying existing code:

1. Follow the existing folder structure by chapter
2. Include both CPU and GPU implementations when applicable
3. Add comments explaining CUDA-specific concepts
4. Test on multiple platforms if possible
5. Document any additional dependencies

## Important Disclaimers

- This code is from the CUDA by Example textbook (2010) and reflects CUDA concepts from that era
- Newer CUDA versions may have deprecated some functions or changed best practices
- Modern CUDA development often uses higher-level libraries (cuDNN, TensorRT, etc.)
- Some graphics examples (Chp05, Chp08) use older OpenGL/GLUT APIs
- Always refer to official NVIDIA documentation for latest CUDA features and best practices

## License

This code is based on examples from "CUDA by Example" by Jason Sanders and Edward Kandrot.
Use and modify according to the book's license terms.

---

**Last Updated:** May 2026
**CUDA Toolkit Tested With:** CUDA 5.0 - 12.0+
**Platforms Supported:** Windows (7+), Linux (Ubuntu, Fedora, etc.), macOS (with limitations)
