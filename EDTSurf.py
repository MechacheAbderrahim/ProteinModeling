import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", type=str, required=False, default="example.pdb", help="PDB Input file name")
parser.add_argument("-t", "--triangulation_type", type=str, required=False, default="2", help="1-MC 2-VCMC (default is 2)")
parser.add_argument("-s", "--surface_type", type=str, required=False, default="3", help="1-VWS 2-SAS 3-MS 0-DEPTH (default is 3)")
parser.add_argument("-c", "--color_mode", type=str, required=False, default="2", help="1-pure 2-atom 3-chain (default is 2)")
parser.add_argument("-p", "--probe_radius", type=str, required=False, default="1.4", help="float point in [0,2.0] (default is 1.4)")
parser.add_argument("-hh", "--surface", type=str, required=False, default="1", help="1-inner and outer 2-outer 3-inner (default is 1)")
parser.add_argument("-f", "--scale_factor", type=str, required=False, default="4.0", help="float point in (0,20.0] (default is 4.0)")
args = parser.parse_args()

input_file = args.input_file
triangulation_type = args.triangulation_type
surface_type = args.surface_type
color_mode = args.color_mode
probe_radius = args.probe_radius
surface = args.surface
scale_factor = args.scale_factor

#  Define paths
edtsurf_exe = os.path.join("bin", "EDTSurf")  # Path to EDTSurf executable
input_path = os.path.join("data/pdb", input_file)  # Input file
output_file = os.path.join("data/ply", input_file[:-4])  # Output file

# Check if EDTSurf exists and is executable
if not os.path.isfile(edtsurf_exe):
    raise FileNotFoundError(f"EDTSurf not found at: {edtsurf_exe}")
if not os.access(edtsurf_exe, os.X_OK):
    print("EDTSurf is not executable. Attempting to fix permissions...")
    os.chmod(edtsurf_exe, 0o755)  # Make it executable

# Construct the command with default parameters (except for -i)
command = [
    edtsurf_exe,  # EDTSurf executable
    "-i", input_path,
    "-o", output_file,
    "-t", triangulation_type,
    "-s", surface_type,
    "-c", color_mode,
    "-p", probe_radius,
    "-h", surface,
    "-f", scale_factor
]
print(command)
# Execute the command
try:
    result = subprocess.run(command, capture_output=True, text=True)

    # Print output and error streams
    print("✅ EDTSurf executed!")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

except Exception as e:
    print(f"❌ Unexpected error: {e}")

# Final check: Verify if the output file was created
if os.path.isfile(output_file+".ply"):
    print("✅ Done with success!")
    print(f"Output file created: {output_file}.ply")
else:
    print("❌ Error: Output file was not generated.")