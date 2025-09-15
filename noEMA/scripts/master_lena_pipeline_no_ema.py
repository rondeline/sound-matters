#!/usr/bin/env python3
"""
master_lena_pipeline_no_ema.py - LENA Analysis Pipeline without EMA files

*** NO-EMA VERSION ***
This is a modified version that processes LENA data without requiring EMA activity files.
Steps 4 and 5 have been removed, and steps 0, 6, and 8 have been modified.

This script orchestrates the LENA analysis pipeline using only recording data (no EMA/activity files).
Removed steps 4 and 5, modified steps 0 and 6 to work without EMA data.

Example usage:
    # Run full pipeline for all subjects
    python master_lena_pipeline_no_ema.py --its_dir "Z:/LENA/its" --output_dir "Z:/Results"
    
    # Run for specific subjects
    python master_lena_pipeline_no_ema.py --subjects S1-S3,S10,S15-S17 --its_dir "Z:/LENA/its" 
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional
import logging
import json
import shutil  
import re 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lena_pipeline_no_ema.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LENAPipelineNoEMA:
    """Master class for running the LENA analysis pipeline without EMA files."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.script_dir = base_dir
        
        # Define script paths - removed steps 4 and 5
        self.scripts = {
            -1: "its_file_renamer.py",
            0: "run_batch_noEMA_analysis.py",  # Modified version
            1: "run_matlab_analysis_combined.py", 
            2: "merge_subject_data.py",
            3: "snr_scatter_plots.py",
            6: "smoothed_snr_plots_no_ema.py",  # Modified version
            7: "FinalPythonScripts/Code_Individual_Data_Percentages_SNRs._UPDATED.py",
            8: "group_average_smooth_plots_no_ema.py"  # Modified version
        }
        
        # Validate all scripts exist
        for step, script in self.scripts.items():
            if step == -1:  # Skip validation for step -1
                continue
            script_path = self.script_dir / script
            if not script_path.exists() and step not in [0, 6, 8]:  # Skip checking for modified scripts
                logger.warning(f"Script not found (will be created): {script_path}")
    
    def parse_subject_range(self, subject_input: str) -> List[str]:
        """Parse subject input supporting ranges and mixed selections."""
        if not subject_input:
            return []
        
        subjects = []
        parts = subject_input.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part and part.count('-') == 1:
                # Handle range like S1-S5
                start_str, end_str = part.split('-')
                start_str = start_str.strip()
                end_str = end_str.strip()
                
                # Extract numbers
                if start_str.startswith('S') and end_str.startswith('S'):
                    try:
                        start_num = int(start_str[1:])
                        end_num = int(end_str[1:])
                        
                        if start_num <= end_num:
                            for i in range(start_num, end_num + 1):
                                subjects.append(f'S{i}')
                        else:
                            logger.warning(f"Invalid range {part} (start > end)")
                    except ValueError:
                        logger.warning(f"Invalid subject format in range {part}")
                else:
                    logger.warning(f"Invalid subject format in range {part}")
            else:
                # Handle individual subject
                if part.startswith('S'):
                    subjects.append(part)
                else:
                    logger.warning(f"Invalid subject format {part}")
        
        return sorted(list(set(subjects)))  # Remove duplicates and sort
    

    def run_step0_batch_analysis_no_ema(self, its_dir: str, subjects: List[str], days: Optional[List[str]] = None) -> bool:
        """Step 0: Run Batch Analysis without EMA files"""
        logger.info("=" * 60)
        logger.info("STEP 0: Batch Analysis (No EMA) - Generate Segment Files")
        logger.info("=" * 60)
        
        # Set up paths
        if its_dir:
            rec_dir = Path(its_dir).parent / "recording_files"
        else:
            # Default to recording_files in the project root (parent of scripts)
            rec_dir = self.script_dir.parent / "recording_files"
        
        output_dir = self.script_dir.parent / "outputs"
        
        cmd = [
            sys.executable,
            str(self.script_dir / self.scripts[0]),
            "--input_dir", str(rec_dir),        # Add this line
            "--output_dir", str(output_dir)     # Add this line
        ]
        
        if subjects:
            cmd.extend(["--subjects", ",".join(subjects)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Batch analysis completed successfully")
            logger.info(result.stdout)
            return self.copy_segment_files(its_dir, subjects)
        except subprocess.CalledProcessError as e:
            logger.error(f"Batch analysis failed: {e.stderr}")
            return False

    def copy_segment_files(self, its_dir: str, subjects: List[str]) -> bool:
        """Copy Excel segment files to its directory with correct naming"""
        logger.info("Copying segment files to its directory...")
        
        segments_dir = self.script_dir.parent / "outputs" / "segments"
        
        # Fix: Handle case where its_dir is not provided - default to 'its' directory
        if not its_dir or its_dir.strip() == "":
            its_path = self.script_dir.parent / "its"
        else:
            its_path = Path(its_dir)
        
        # Create the its directory if it doesn't exist
        its_path.mkdir(parents=True, exist_ok=True)
        
        if not segments_dir.exists():
            logger.warning(f"Segments directory not found: {segments_dir}")
            return True  # Not a fatal error
        
        # Find all Excel files in segments directory
        excel_files = list(segments_dir.glob("*.xlsx"))
        logger.info(f"Found {len(excel_files)} Excel files in segments directory")
        
        copied_count = 0
        for excel_file in excel_files:
            # Parse filename to extract subject and day
            filename = excel_file.stem
            match = re.match(r'(S\d+)_Day(\d+)_segments', filename)
            
            if match:
                subject = match.group(1)
                day = match.group(2)
                
                # Skip if filtering subjects and this subject not in list
                if subjects and subject not in subjects:
                    continue
                    
                # Create new filename: S##_r#.xlsx
                new_filename = f"{subject}_r{day}.xlsx"
                destination = its_path / new_filename
                
                try:
                    shutil.copy2(excel_file, destination)
                    logger.info(f"Copied {excel_file.name} -> {new_filename} to {its_path}")
                    copied_count += 1
                except Exception as e:
                    logger.error(f"Failed to copy {excel_file.name}: {e}")
                    return False
        
        logger.info(f"Copied {copied_count} segment files to {its_path}")
        return True

    def run_step6_smoothed_snr_plots(self, subjects: List[str], days: Optional[List[str]] = None, window_minutes: int = 15) -> bool:
        """Step 6: Generate Smoothed SNR Plots (without activities)"""
        logger.info("=" * 60)
        logger.info("STEP 6: Smoothed SNR Plot Generation (No Activities)")
        logger.info("=" * 60)
        
        cmd = [sys.executable, str(self.script_dir / self.scripts[6])]
        
        if subjects:
            cmd.extend(["--subjects"] + subjects)
        
        if days:
            cmd.extend(["--days"] + days)
        
        cmd.extend(["--window-minutes", str(window_minutes)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Smoothed SNR plots completed successfully")
            logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Smoothed SNR plot generation failed: {e.stderr}")
            return False
    
    def run_pipeline(self, its_dir: str, output_dir: str, subjects: List[str], 
                days: Optional[List[str]] = None, steps: List[int] = None,
                matlab_path: Optional[str] = None, matlab_version: str = "5.0.1") -> bool:
        """Run the complete pipeline with specified steps"""
        
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("LENA ANALYSIS PIPELINE (NO EMA) STARTING")
        logger.info("=" * 60)
        logger.info(f"ITS Directory: {its_dir}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info(f"Subjects: {subjects if subjects else 'All'}")
        logger.info(f"Days: {days if days else 'All'}")
        logger.info(f"Steps to run: {steps}")
        logger.info("=" * 60)
        
        # Default to all steps except 4 and 5
        if steps is None:
            steps = [-1, 0, 1, 2, 3, 6, 7, 8]
        
        # Step -1: ITS file renaming (if requested)
        if -1 in steps:
            success = self.run_step_minus1_its_rename(its_dir, subjects)
            if not success:
                logger.error("Pipeline failed at Step -1 (ITS Rename)")
                return False
        
        # Step 0: Batch analysis (no EMA)
        if 0 in steps:
            success = self.run_step0_batch_analysis_no_ema(its_dir, subjects, days)
            if not success:
                logger.error("Pipeline failed at Step 0 (Batch Analysis)")
                return False
        
        # Step 1: MATLAB analysis
        if 1 in steps:
            success = self.run_step1_matlab(its_dir, output_dir, subjects, matlab_path)
            if not success:
                logger.error("Pipeline failed at Step 1 (MATLAB Analysis)")
                return False

        # Step 2: Merge data
        if 2 in steps:
            success = self.run_step2_merge_data(output_dir, subjects, matlab_version)
            if not success:
                logger.error("Pipeline failed at Step 2 (Merge Data)")
                return False

        # Step 3: SNR plots
        if 3 in steps:
            success = self.run_step3_snr_plots(subjects, days)
            if not success:
                logger.error("Pipeline failed at Step 3 (SNR Plots)")
                return False

        # Step 6: Smoothed SNR Plots (no activities)
        if 6 in steps:
            success = self.run_step6_smoothed_snr_plots(subjects, days)
            if not success:
                logger.error("Pipeline failed at Step 6 (Smoothed SNR Plots)")
                return False

        # Step 7: Individual data percentages
        if 7 in steps:
            success = self.run_step7_individual_percentages(subjects)
            if not success:
                logger.error("Pipeline failed at Step 7 (Individual Percentages)")
                return False

        # Step 8: Group average plots
        if 8 in steps:
            success = self.run_step8_group_averages(subjects)
            if not success:
                logger.error("Pipeline failed at Step 8 (Group Averages)")
                return False
        
        # Pipeline completed
        elapsed_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total time: {elapsed_time/60:.2f} minutes")
        logger.info("=" * 60)
        
        return True
    
    # Include the other step methods from the original that don't need modification
    def run_step_minus1_its_rename(self, its_dir: str, subjects: List[str]) -> bool:
        """Step -1: Rename ITS files to match expected format"""
        logger.info("=" * 60)
        logger.info("STEP -1: ITS File Renaming")
        logger.info("=" * 60)
        
        its_path = Path(its_dir)
        renamed_count = 0
        
        for its_file in its_path.glob("*.its"):
            # Try to extract subject and day from filename
            match = re.match(r'.*[Ss](\d+).*[Dd]ay(\d+)', its_file.stem)
            if match:
                subject_num = match.group(1)
                day = match.group(2)
                new_name = f"s{subject_num}d{day}.its"
                
                if subjects and f"S{subject_num}" not in subjects:
                    continue
                    
                if its_file.name != new_name:
                    new_path = its_file.parent / new_name
                    logger.info(f"Renaming: {its_file.name} -> {new_name}")
                    its_file.rename(new_path)
                    renamed_count += 1
        
        logger.info(f"Renamed {renamed_count} ITS files")
        return True
    
    def run_step1_matlab(self, its_dir: str, output_dir: str, subjects: List[str], matlab_path: Optional[str] = None) -> bool:
        """Step 1: Run MATLAB Analysis"""
        logger.info("=" * 60)
        logger.info("STEP 1: MATLAB Analysis")
        logger.info("=" * 60)
        
        cmd = [sys.executable, str(self.script_dir / self.scripts[1]), "--its_dir", its_dir, "--output_dir", output_dir]
        
        if matlab_path:
            cmd.extend(["--matlab_path", matlab_path])
        
        if subjects:
            cmd.extend(["--subject", ",".join(subjects)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Remove capture_output=True to allow real-time output display
            result = subprocess.run(cmd, check=True, text=True)
            logger.info("MATLAB analysis completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"MATLAB analysis failed with return code: {e.returncode}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("MATLAB analysis timed out after 10 minutes")
            return False

    def run_step2_merge_data(self, output_dir: str, subjects: List[str], matlab_version: str = "5.0.1") -> bool:
        """Step 2: Merge Subject Data into Master File"""
        logger.info("=" * 60)
        logger.info("STEP 2: Merge Subject Data")
        logger.info("=" * 60)
        
        # Import and call the merge function directly
        try:
            sys.path.append(str(self.script_dir))
            from merge_subject_data import merge_subject_data
            
            from merge_subject_data import get_version_file_names
            individual_name, batch_name = get_version_file_names(matlab_version)
            source_file = Path(output_dir) / f"version_{matlab_version}" / individual_name
            target_file = self.script_dir / "FinalPythonScripts" / "Individual_SNR_Power_NIH.xlsx"
            
            logger.info(f"Merging data from: {source_file}")
            logger.info(f"Into master file: {target_file}")
            
            success = merge_subject_data(source_file, target_file, dry_run=False, version=matlab_version)
            
            if success:
                logger.info("Subject data merge completed successfully")
                return True
            else:
                logger.error("Subject data merge failed")
                return False
                
        except Exception as e:
            logger.error(f"Merge data step failed: {e}")
            return False
    
    def run_step3_snr_plots(self, subjects: List[str], days: Optional[List[str]] = None) -> bool:
        """Step 3: Generate SNR Scatter Plots"""
        logger.info("=" * 60)
        logger.info("STEP 3: SNR Scatter Plot Generation")
        logger.info("=" * 60)
        
        cmd = [sys.executable, str(self.script_dir / self.scripts[3])]
        
        if subjects:
            cmd.extend(["--subjects", ",".join(subjects)])
        
        if days:
            cmd.extend(["--days"] + days)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("SNR scatter plots completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"SNR plot generation failed: {e.stderr}")
            return False
    
    def run_step7_individual_percentages(self, subjects: List[str]) -> bool:
        """Step 7: Generate Individual Data Percentages and Histograms"""
        logger.info("=" * 60)
        logger.info("STEP 7: Individual Data Percentages")
        logger.info("=" * 60)
        
        file_to_read = self.script_dir / "FinalPythonScripts" / "Individual_SNR_Power_NIH.xlsx"
        
        cmd = [
            sys.executable,
            str(self.script_dir / self.scripts[7]),
            str(file_to_read)
        ]
        
        if subjects:
            for subject in subjects:
                cmd.extend(["--subject", subject])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Individual percentages completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Individual percentages failed: {e.stderr}")
            return False
    
    def run_step8_group_averages(self, subjects: List[str]) -> bool:
        """Step 8: Generate Group Average Plots (No Activities)"""
        logger.info("=" * 60)
        logger.info("STEP 8: Group Average Plot Generation (No Activities)")
        logger.info("=" * 60)
        
        cmd = [sys.executable, str(self.script_dir / self.scripts[8])]
        
        if subjects:
            cmd.extend(["--subjects", ",".join(subjects)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Group average plots completed successfully")
            if result.stdout:
                logger.info(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Group average plot generation failed: {e.stderr}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="LENA Analysis Pipeline (No EMA Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Pipeline Steps (removed 4 and 5):
    -1: Rename ITS files
     0: Batch analysis (no EMA) - Generate segment files from recordings only
     1: MATLAB analysis - Process ITS files
     2: Merge data - Combine MATLAB outputs
     3: SNR plots - Generate SNR scatter plots
     6: Smoothed plots - Generate smoothed SNR plots (no activities)
     7: Individual percentages - Generate histograms and statistics
     8: Group averages - Generate group average plots
            """
    )
    
    # Required arguments
    parser.add_argument("--its_dir", 
                       help="Directory containing .its and .xlsx files (required for Step 1)")
    
    # Optional arguments
    parser.add_argument("--output_dir", default="./matlab_outputs",
                       help="Base output directory (default: ./matlab_outputs)")
    
    parser.add_argument("--subjects", "-s",
                       help="Subjects to process. Supports ranges and mixed selection (e.g., 'S1-S3,S10,S15-S17')")
    
    parser.add_argument("--days", "-d",
                       help="Days to process (e.g., '1,2' or '1-3')")
    
    parser.add_argument("--steps", default="0,1,2,3,6,7,8",
                       help="Pipeline steps to run (default: all except 4,5)")
    
    parser.add_argument("--matlab-path",
                       help="Path to MATLAB executable (if not in PATH)")
    parser.add_argument("--matlab_version", 
                   choices=["5.0.1", "4.6", "4.3.1"],
                   default="5.0.1",
                   help="MATLAB version to use (determines output file naming)")
    
    args = parser.parse_args()
    
    # Parse inputs
    if args.its_dir:
        its_dir = Path(args.its_dir)
        if not its_dir.exists():
            logger.error(f"ITS directory not found: {its_dir}")
            sys.exit(1)
    else:
        its_dir = None
    
    output_dir = Path(args.output_dir)
    script_dir = Path(__file__).parent
    
    # Initialize pipeline
    pipeline = LENAPipelineNoEMA(script_dir)
    
    # Parse subjects
    subjects = pipeline.parse_subject_range(args.subjects) if args.subjects else []
    
    # Parse days
    if args.days:
        days = []
        for day_part in args.days.split(','):
            if '-' in day_part:
                start, end = day_part.split('-')
                days.extend([str(d) for d in range(int(start), int(end) + 1)])
            else:
                days.append(day_part)
    else:
        days = None
    
    # Parse steps
    steps = [int(s) for s in args.steps.split(',')]
    
    # Validate required arguments for specific steps
    if 1 in steps and not its_dir:
        logger.error("--its_dir is required for Step 1 (MATLAB analysis)")
        sys.exit(1)
    
    # Run pipeline
    success = pipeline.run_pipeline(
        its_dir=str(its_dir) if its_dir else "",
        output_dir=str(output_dir),
        subjects=subjects,
        days=days,
        steps=steps,
        matlab_path=args.matlab_path,
        matlab_version=args.matlab_version
    )

    
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()