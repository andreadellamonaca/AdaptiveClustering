# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/andrea/Scrivania/clion-2019.2.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/andrea/Scrivania/clion-2019.2.2/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/andrea/Scrivania/AdaptiveClustering

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andrea/Scrivania/AdaptiveClustering/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/AdaptiveClustering.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/AdaptiveClustering.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/AdaptiveClustering.dir/flags.make

CMakeFiles/AdaptiveClustering.dir/main.cpp.o: CMakeFiles/AdaptiveClustering.dir/flags.make
CMakeFiles/AdaptiveClustering.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andrea/Scrivania/AdaptiveClustering/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/AdaptiveClustering.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/AdaptiveClustering.dir/main.cpp.o -c /home/andrea/Scrivania/AdaptiveClustering/main.cpp

CMakeFiles/AdaptiveClustering.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/AdaptiveClustering.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/andrea/Scrivania/AdaptiveClustering/main.cpp > CMakeFiles/AdaptiveClustering.dir/main.cpp.i

CMakeFiles/AdaptiveClustering.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/AdaptiveClustering.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/andrea/Scrivania/AdaptiveClustering/main.cpp -o CMakeFiles/AdaptiveClustering.dir/main.cpp.s

CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.o: CMakeFiles/AdaptiveClustering.dir/flags.make
CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.o: ../adaptive_clustering.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andrea/Scrivania/AdaptiveClustering/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.o -c /home/andrea/Scrivania/AdaptiveClustering/adaptive_clustering.cpp

CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/andrea/Scrivania/AdaptiveClustering/adaptive_clustering.cpp > CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.i

CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/andrea/Scrivania/AdaptiveClustering/adaptive_clustering.cpp -o CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.s

CMakeFiles/AdaptiveClustering.dir/error.cpp.o: CMakeFiles/AdaptiveClustering.dir/flags.make
CMakeFiles/AdaptiveClustering.dir/error.cpp.o: ../error.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andrea/Scrivania/AdaptiveClustering/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/AdaptiveClustering.dir/error.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/AdaptiveClustering.dir/error.cpp.o -c /home/andrea/Scrivania/AdaptiveClustering/error.cpp

CMakeFiles/AdaptiveClustering.dir/error.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/AdaptiveClustering.dir/error.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/andrea/Scrivania/AdaptiveClustering/error.cpp > CMakeFiles/AdaptiveClustering.dir/error.cpp.i

CMakeFiles/AdaptiveClustering.dir/error.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/AdaptiveClustering.dir/error.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/andrea/Scrivania/AdaptiveClustering/error.cpp -o CMakeFiles/AdaptiveClustering.dir/error.cpp.s

# Object files for target AdaptiveClustering
AdaptiveClustering_OBJECTS = \
"CMakeFiles/AdaptiveClustering.dir/main.cpp.o" \
"CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.o" \
"CMakeFiles/AdaptiveClustering.dir/error.cpp.o"

# External object files for target AdaptiveClustering
AdaptiveClustering_EXTERNAL_OBJECTS =

AdaptiveClustering: CMakeFiles/AdaptiveClustering.dir/main.cpp.o
AdaptiveClustering: CMakeFiles/AdaptiveClustering.dir/adaptive_clustering.cpp.o
AdaptiveClustering: CMakeFiles/AdaptiveClustering.dir/error.cpp.o
AdaptiveClustering: CMakeFiles/AdaptiveClustering.dir/build.make
AdaptiveClustering: /usr/lib/x86_64-linux-gnu/libarmadillo.so
AdaptiveClustering: CMakeFiles/AdaptiveClustering.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/andrea/Scrivania/AdaptiveClustering/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable AdaptiveClustering"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/AdaptiveClustering.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/AdaptiveClustering.dir/build: AdaptiveClustering

.PHONY : CMakeFiles/AdaptiveClustering.dir/build

CMakeFiles/AdaptiveClustering.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/AdaptiveClustering.dir/cmake_clean.cmake
.PHONY : CMakeFiles/AdaptiveClustering.dir/clean

CMakeFiles/AdaptiveClustering.dir/depend:
	cd /home/andrea/Scrivania/AdaptiveClustering/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andrea/Scrivania/AdaptiveClustering /home/andrea/Scrivania/AdaptiveClustering /home/andrea/Scrivania/AdaptiveClustering/cmake-build-debug /home/andrea/Scrivania/AdaptiveClustering/cmake-build-debug /home/andrea/Scrivania/AdaptiveClustering/cmake-build-debug/CMakeFiles/AdaptiveClustering.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/AdaptiveClustering.dir/depend

