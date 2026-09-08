
import re
from shutil import copyfile
import subprocess
from sgis._utils import get_logger

class SplittingRecap:
    """
    A class to recap and verify the results of a building image splitting process.
    This class collects statistics about the splitting operation, including counts of generated
    JPG images, builds verification notes, and validates that the output matches expectations.
    """
  
    # dossier de sortie contenant les résultats du splitter
    output_folder_path : str
    preprocessing_step_log_file : str
    final_notes_file : str
    
    # compteurs 
    initial_buildings_count : int
    unwanted_buidings_count : int
    jpg_count: int
    underscore_jpg_count : int
    small_jpg_count : int
    progress_file_lines_count: int
    rasters_count: int
    rasters_containing_buildings_count: int
    
    # commandes os pour comptage des fichiers
    jpg_count_cmd : str
    underscore_jpg_count_cmd: str
    small_jpg_count_cmd : str
    progress_file_lines_count_cmd: str
    file_count_cmd : str
    
    def __init__(self, output_folder_path):
        self.output_folder_path = output_folder_path
        
        # setting folders paths
        rasters_fp = self.output_folder_path  / "rasters"
        images_fp = rasters_fp / "images"
        repartition_fp = rasters_fp / "repartition_by_rasters"
        preprocessing_fp = self.output_folder_path  / "preprocessing" 
        self.preprocessing_step_log_file = preprocessing_fp / "vectors/preprocessing.log" 
        self.final_notes_file = self.output_folder_path  / "notes.txt"
        
        self.jpg_count_cmd = rf'find {images_fp} -iname "*.jpg" | wc -l'
        self.underscore_jpg_count_cmd = rf'find {images_fp} -iname "*_*.jpg" | wc -l'
        self.small_jpg_count_cmd = rf'find {images_fp} -iname "*.jpg" -size -100c | wc -l'
        self.progress_file_lines_count_cmd = rf'cat {rasters_fp}/progress.txt | wc -l'
        self.file_count_cmd = rf'ls {repartition_fp} | wc -l'
    
    def summarize(self):
        logger = get_logger()
        with open(self.preprocessing_step_log_file, "r", encoding="utf-8") as f:
            log_content = f.read()
            self._parse_preprocessing_log(log_content)
        
        self._count_files()
        self._check_counts()
        self._write_notes_file()
        
        logger.info("Open file for summarizing the results of splitting: file://%s", self.final_notes_file)
    
    def _parse_preprocessing_log(self, log_file_text : str):
        """
        Parses the preprocessing.log text to find and extract the number of buildings
        that were removed (smaller than 10m²) and the initial total building count.
        """
        # decode preprocessing.log
        match = re.search(r"(\d+)\s+buildings out of\s+(\d+)", log_file_text) # exemple : "11173 buildings out of 244658"
        if match:
            self.unwanted_buidings_count = int(match.group(1))
            self.initial_buildings_count = int(match.group(2))
        else :
            raise AttributeError("Expected line not found: \n - X buildings out of Y, smaller than 10m², removed")
  
    def _count_files(self):
        """
        Count various file types and progress file lines in the splitting process.
        Executes multiple commands to count:
        - jpg_count: Number of JPG files
        - underscore_jpg_count: Number of JPG files with underscore naming convention
        Etecetera
        """
        
        self.jpg_count = self._run(self.jpg_count_cmd)
        self.underscore_jpg_count = self._run(self.underscore_jpg_count_cmd)
        self.small_jpg_count = self._run(self.small_jpg_count_cmd)
        self.progress_file_lines_count = self._run(self.progress_file_lines_count_cmd)
        self.file_count = self._run(self.file_count_cmd)
        
    def _write_notes_file(self):
        
        # recopie contenu du fichier de log de l'étape de preprocessing > notes.txt
        copyfile(self.preprocessing_step_log_file, self.final_notes_file)
        
        # Edit files "notes"
        count_lines =   (
                            f"\n\n{self.jpg_count_cmd}\n{self.jpg_count}\n"
                            f"{self.underscore_jpg_count_cmd}\n{self.underscore_jpg_count}\n"
                            f"{self.small_jpg_count_cmd}\n{self.small_jpg_count}\n"
                            f"{self.progress_file_lines_count_cmd}\n{self.progress_file_lines_count}\n"
                            f"{self.file_count_cmd}\n{self.file_count}\n\n\n"
                        )
                        
        additional  =   (
                            f"# ls ~/temporary_LaCie/rasters/only_tiles/[...]/*jp2 | wc -l\n"
                            f"# find rasters/images -iname \"*.jpg\" -size -100c -delete\n\n\n"
                        )
                        
        end_notes   =   (
                            f"Verification:\n"
                            f"- nombre théorique : {self.initial_buildings_count} - {self.unwanted_buidings_count} =  {self.initial_buildings_count -  self.unwanted_buidings_count}\n"
                            f"- nombre obtenu : {self.jpg_count} - {self.underscore_jpg_count} =  {self.jpg_count - self.underscore_jpg_count}\n"
                        )
        with open(self.final_notes_file, "a", encoding="utf-8") as f:
            f.write(count_lines + additional + end_notes)
            
    def _build_notes_content(self):
        """
        Builds the content for the notes file, summarizing the counts and verification results.
        Returns a string containing the formatted notes.
        """
        # recopie contenu du fichier de log de l'étape de preprocessing > notes.txt
        copyfile(self.preprocessing_step_log_file, self.final_notes_file)
        
        count_lines =   (
                            f"\n\n{self.jpg_count_cmd}\n{self.jpg_count}\n"
                            f"{self.underscore_jpg_count_cmd}\n{self.underscore_jpg_count}\n"
                            f"{self.small_jpg_count_cmd}\n{self.small_jpg_count}\n"
                            f"{self.progress_file_lines_count_cmd}\n{self.progress_file_lines_count}\n"
                            f"{self.file_count_cmd}\n{self.file_count}\n\n\n"
                        )
                        
        additional  =   (
                            f"# ls ~/temporary_LaCie/rasters/only_tiles/[...]/*jp2 | wc -l\n"
                            f"# find rasters/images -iname \"*.jpg\" -size -100c -delete\n\n\n"
                        )
                        
        end_notes   =   (
                            f"Verification:\n"
                            f"- nombre théorique : {self.initial_buildings_count} - {self.unwanted_buidings_count} =  {self.initial_buildings_count -  self.unwanted_buidings_count}\n"
                            f"- nombre obtenu : {self.jpg_count} - {self.underscore_jpg_count} =  {self.jpg_count - self.underscore_jpg_count}\n"
                        )
        return count_lines + additional + end_notes    
        
        
    def _check_counts(self):
        logger = get_logger()
        # Checks images counts:
        theoretical_number  = self.initial_buildings_count -  self.unwanted_buidings_count
        obtained_number =  self.jpg_count - self.underscore_jpg_count
        images_diff = theoretical_number - obtained_number
        if(images_diff == 0):
            logger.info("Number of buildings obtained after splitting is OK")
        else:
            logger.warning(self._build_notes_content())
            if (self.small_jpg_count != 0):
                logger.warning(f"Some images with near-zero disk usage exist: {self.small_jpg_count}")
            raise AssertionError("Number of buildings obtained after splitting is incoherent")
            
    def _run(self, cmd):
            return int(subprocess.check_output(cmd, shell=True, text=True).strip())