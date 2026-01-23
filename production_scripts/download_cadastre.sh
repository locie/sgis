#! /bin/sh


# Download a compressed version of the Etalab cadastre for a given department, and extract it.
# Data is stored on Lacie. 
# 
# ex: sh download_cadastre.sh --dep 42 --date 2004-04-01
#
# Date format must be: YYYY-MM-JJ
# Available month ("MM") must be checked on: https://cadastre.data.gouv.fr/datasets/cadastre-etalab


	
# check that LaCie is connected
if [ ! -d $HOME/LaCie_thebaulm/gis/vectors/cadastre ]
then
	echo "$HOME/LaCie_thebaulm/gis/vectors/cadastre not found"
	exit
fi

date=""
dep=""


# Set options for the getopt command
options=$(getopt -o "nag" -l "date:,dep:" -- "$@")
if [ $? -ne 0 ]; then
    echo "Invalid arguments."
    exit 1
fi
eval set -- "$options"

# Read the named argument values
while [ $# -gt 0 ]; do
    case "$1" in
        --date) date="$2"; shift;;
        --dep) dep="$2"; shift;;
    esac
    shift
done

if [ -z "$date" ]; then
	echo "Please provide the 'date' argument ('--date YYYY-MM-JJ')"
	exit
fi

if [ -z $dep ]; then
	echo "Please provide the 'dep' argument ('--dep XX')"
	exit
fi

# Check that department is 2 digits for 01 to 99 (0 padded) and 3 digits for DOM TOM
nbr_digits=$(echo -n $dep | wc -c)
if [ $nbr_digits -eq 1 ]
then
	dep=0$dep
elif [ $nbr_digits -eq 3 ]
then
	if [ $(echo $dep | head -c 1) -eq "0" ]
	then
		dep=$(echo -n $dep | tail -c 2)
	fi	
fi

echo "Department: $dep"


rootp="$HOME/LaCie_thebaulm/gis/vectors/cadastre/$date"
filename="cadastre-$dep-batiments-shp"

# check that data does not already exists in LaCie
if [ -d $HOME/LaCie_thebaulm/gis/vectors/cadastre/$date/unzipped/$filename ]
then
	echo "Data exists in $HOME/LaCie_thebaulm/gis/vectors/cadastre/$date/unzipped/$filename"
	echo "Use it or remove the directory"
	exit
fi



mkdir -p "$rootp/zipped"
mkdir -p "$rootp/unzipped"


URL="https://cadastre.data.gouv.fr/data/etalab-cadastre/$date/shp/departements/$dep/$filename.zip"
wget $URL --directory-prefix "$rootp/zipped"

if [ $? -ne 0 ]
then
	echo "Failed to download file:"
	echo "$URL"
	exit
fi

unzip "$rootp/zipped/$filename.zip" -d "$rootp/unzipped/$filename"

if [ $? -ne 0 ]
then
	echo "Failed to extract content of:"
	echo "	$rootp/zipped/$filename.zip"
	echo "in:"
	echo "	$rootp/unzipped/$filename"
	exit
fi
