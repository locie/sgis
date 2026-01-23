#!/bin/sh

# Display a given building image knowing its ID, department and year.

# ex: sh display.sh --dep 75 --year 2011 --img_name 0755654



# Set options for the getopt command
options=$(getopt -o "nag" -l "dep:,year:,img_name:" -- "$@")
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
        --img_name) img_name="$2"; shift;;
    esac
    shift
done



# remove extension if any
img_name="${img_name%.*}"

# add the extension
img_name=${img_name}.jpg

# check that first img_name digits correspond to department code 
# nbr_dep_digits=$(echo -n $dep | wc -m)
# first_digits=$(echo -n ${img_name} | head -c ${nbr_dep_digits})


# remove leading 0 if any
# img_name="${img_name##*(0)}"


if [ -f ~/split/$dep/$year/rasters/images/$img_name ]
then
	display ~/split/$dep/$year/rasters/images/$img_name	
else
	echo "$img_name not found"
	
	if [ -f ~/split/$dep/$year/rasters/images/0$img_name ]
	then
		display ~/split/$dep/$year/rasters/images/0$img_name	
	else
		echo "0$img_name not found"
		
		if [ -f ~/split/$dep/$year/rasters/images/00$img_name ]
		then
			display ~/split/$dep/$year/rasters/images/00$img_name
		else
			echo "==>Image does not exist"
		fi
	fi
fi
