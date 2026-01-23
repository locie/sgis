#! /bin/sh



# Download a compressed version of the RGB IGN BDOrtho database for a given department, and extract it.
# Metadata is stored on Lacie, extracted raster tiles are stored locally.
# technical notes: 
#
# - for a given year, each department comes in several 7z parts
# - this script makes strong assumptions on the naming convention of each 7z file
#
# ex: sh download_rasters.sh --dep 56 --year 2023
# ex: sh download_rasters.sh --dep 92 --year 2021 --resolution 15
#
# The version of BDOrtho is specified by the 'year' argument.
# The default resolution is 20, i.e. 20 cm/pixel


if [ ! -d $HOME/temporary_LaCie/rasters/only_tiles ]
then
	echo "$HOME/temporary_LaCie not found"
	exit
fi
	

if [ ! -d $HOME/LaCie_thebaulm/gis/rasters/archives ]
then
	echo "$HOME/LaCie_thebaulm/gis/rasters/archives not found"
	exit
fi

year=""
dep=""
resolution=""


# Set options for the getopt command
options=$(getopt -o "nag" -l "year:,dep:,resolution:" -- "$@")
if [ $? -ne 0 ]; then
    echo "Invalid arguments."
    exit 1
fi
eval set -- "$options"

# Read the named argument values
while [ $# -gt 0 ]; do
    case "$1" in
        --year) year="$2"; shift;;
        --dep) dep="$2"; shift;;
        --resolution) resolution="$2"; shift;;
    esac
    shift
done

if [ -z $year ]; then
	echo "Please provide the 'year' argument ('--year XXXX')"
	exit
fi

if [ -z $dep ]; then
	echo "Please provide the 'dep' argument ('--dep XX')"
	exit
fi

if [ -z $resolution ]; then
	echo "Resolution was not set, assuming 20 cm"
	resolution=20
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


# check that data does not already exists in the archive directory of LaCie
if [ -d $HOME/LaCie_thebaulm/gis/rasters/archives/$dep/$year ]
then
	echo -n "Data exists in $HOME/LaCie_thebaulm/gis/rasters/archives/$dep/$year"
	if [ -d $HOME/LaCie_thebaulm/gis/rasters/only_tiles/$dep/$year ]
	then
		echo " and in $HOME/LaCie_thebaulm/gis/rasters/only_tiles/$dep/$year"
	else
		if [ -d $HOME/temporary_LaCie/rasters/only_tiles/$dep/$year ]
		then
			echo " and in $HOME/temporary_LaCie/rasters/only_tiles/$dep/$year"
		else
			echo " but no data found in $HOME/temporary_LaCie/rasters/only_tiles/$dep/$year"
			echo "Check for data in 'temporary_LaCie' on the other platform"
		fi
	fi
	exit
fi




echo "Getting all relevant URLs"
bdortho=$(wget -qO- https://geoservices.ign.fr/bdortho)
relevant=$(echo "$bdortho" | grep -Po '(?<=href=")[^"]*' | grep 7z | grep RVB | grep -E "D0?${dep}_${year}" | grep -E "0M${resolution}")



# check that at least one URL has been found
echo ""
if [ -z "${relevant}" ]
then
	echo "No URL found, please make sure the data you are looking for exists."
	exit
else
	echo  "Found these files:"
	echo "${relevant}"
fi


echo ""
if [ $(echo "${relevant}" | wc -l) -gt 1 ]
then
	echo "Please make sure that these files correspond only to the department/year/resolution you are looking for."
fi
echo ""
echo "Download will start in 10 seconds"


sleep 10
echo ""


 
# attempt 3 downloads
for attempt in 1 2 3
do
	for url in $(echo $relevant)
	do
		part=$(echo "$url" | awk -F\/ ' { print $NF } ')
		if [ -f $HOME/temporary_LaCie/rasters/downloads.temp/$dep/$year/$part  ]
		then
			echo "Skipping download of part $part since"
			echo "File exists in: $HOME/temporary_LaCie/rasters/downloads.temp/$dep/$year"
		else
			mkdir -p "$HOME/temporary_LaCie/rasters/downloads.temp/$dep/$year/"
			wget "$url" --directory-prefix=$HOME/temporary_LaCie/rasters/downloads.temp/$dep/$year/
		fi
	done
done	
	
	
	
# extract	
part1=$(echo "$relevant" | head -n 1 | awk -F\/ ' { print $NF } ')

## determine the name of extraction
extracted=$(7z l $HOME/temporary_LaCie/rasters/downloads.temp/$dep/$year/$part1 -ba | tail -n 1 | awk -F' ' '{print $NF}' | awk -F'/' ' { print $1 } ')
extracted=$HOME/temporary_LaCie/rasters/downloads.temp/$dep/$year/$extracted

if [ -d $extracted ]
then 
	echo "Files exist, aborting extraction"
else
	7z x $HOME/temporary_LaCie/rasters/downloads.temp/$dep/$year/$part1 -o$HOME/temporary_LaCie/rasters/downloads.temp/$dep/$year/
	if [ $? -ne 0 ]
	then
		echo "Unzip error"
		echo "Clean your downloads in $HOME/temporary_LaCie/rasters/downloads.temp/$dep/$year and start again"
		exit
	else
		echo "Unzip done"
	fi
fi


# move tiles
tiles=$(find $extracted -iname "$dep-$year*.tab" -o -iname "$dep-$year*.jp2" -o -iname "$dep-$year*.jp2.aux.xml")


first_tile=$(echo "$tiles" | head -n 1)
first_tile=$(basename -- "$first_tile")
resolution2=$(echo "$first_tile" | awk -F'-' '{ print $6}')



if [ "$resolution2" !=  "0M$resolution" ]
then
	echo "Resolution mismatch: "
	echo "Found $resolution2 from ${first_tile}, but got ${resolution} as an argument."
fi


# echo "$tiles" 

mkdir -p $HOME/temporary_LaCie/rasters/only_tiles/$dep/$year/$dep-$year-0M$resolution-RGB
mv $tiles $HOME/temporary_LaCie/rasters/only_tiles/$dep/$year/$dep-$year-0M$resolution-RGB



if [ $? -ne 0 ]
then
	for tile in $tiles
	do
		mv "$tile" $HOME/temporary_LaCie/rasters/only_tiles/$dep/$year/$dep-$year-0M$resolution-RGB
	done
	if [ $? -ne 0 ]
	then
		echo "Failed to move files from $extracted to $HOME/temporary_LaCie/rasters/only_tiles/$dep/$year/$dep-$year-0M$resolution-RGB"
		exit
	fi
fi


s=$(du -sm $extracted | awk '{print $1}')
echo $s

if [ $s -gt 5 ] 
then
	echo "Archive part of the rasters might be too big. Check that no tile exists in $HOME/LaCie_thebaulm/gis/rasters/archives/$dep/$year/"
fi

mkdir -p $HOME/LaCie_thebaulm/gis/rasters/archives/$dep/$year
mv $extracted $HOME/LaCie_thebaulm/gis/rasters/archives/$dep/$year/ 2> /dev/null

if [ $? -eq 0 ]
then
	echo "Download and move terminated. You can delete the corresponding 7z files in $HOME/temporary_LaCie/rasters/downloads.temp/$dep/$year."
else
	echo "Failed to move files"
	exit
fi

