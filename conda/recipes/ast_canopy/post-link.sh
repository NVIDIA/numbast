# Copyright (c) 2024, NVIDIA CORPORATION.

#!/bin/bash

echo "Copying libastcanopy to $PREFIX/lib" >> $PREFIX/.messages.txt
cp libastcanopy.so $PREFIX/lib
