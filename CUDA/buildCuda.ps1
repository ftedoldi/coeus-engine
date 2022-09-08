python .\CUDA\cudaFindAndBuild.py

cd CUDA;

foreach ($line in ls) {
    $line = $line.toString();

    if ($line.Substring($line.length - 2) -eq "cu") {
        $name, $extension = $line.Split(".");
        nvcc -o Shared/$name.dll --shared $line;

        mv Shared/$name.dll ../Build;
        mv Shared/$name.lib ../Includes/Libraries;
    }
}

cd ..;