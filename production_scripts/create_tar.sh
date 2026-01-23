#! /bin/bash
#
# Create tar archives from split images of a given department in order to upload them on
# recherche.data.gouv.fr.
# Each archive contains 20000 files by default, i.e. 10000 images and their 10000 metadata files.
#
# ex: sh create_tar.sh --dep 75 --year 2011 --card=20000
# "--card" is optional

# Set options for the getopt command
options=$(getopt -o "nag" -l "dep:,year:,card::" -- "$@")
if [ $? -ne 0 ]; then
    echo "Invalid arguments."
    exit 1
fi
eval set -- "$options"

# Read the named argument values
while [ $# -gt 0 ]; do
    case "$1" in
        --dep) dep="$2"; shift;;
        --year) year="$2"; shift;;
        --card) card="$2"; shift;;

    esac
    shift
done


if [ -z "$card" ]
then
	echo -n "Using default cardinal value: "
	card=20000
else
	echo -n "Using given cardinal value: "
	
fi
echo $card

images_parent_dir=$HOME/split/$dep/$year/rasters/

current="$pwd"
cd "$images_parent_dir"

mkdir -p compressed_images/images_repartition

find images -type f | sort > compressed_images/images_repartition/find_results
cd compressed_images/images_repartition

split -l $card -d find_results images_
cd ../..
for file in $(ls compressed_images/images_repartition/images_*)
do 
	echo  ""
	echo "Processing: $file"
	time -f "	Running time: %U user %S system %E elapsed" tar -czf $file.tar.gz -T $file 
	echo -n "	Verification files number:"
	tar --list -f $file.tar.gz | wc -l
done


mkdir -p compressed_images/tar_files
mv compressed_images/images_repartition/images_*.tar.gz compressed_images/tar_files

cd "$current"
#time cat $file |  xargs cp -t "$nbr";time tar -czf $nbr.tar.gz $nbr
