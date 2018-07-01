
# ls . | xargs -I %  unzip -d "../input_dirs/%" %
# find . -type f -name *.zip  | grep -E "[Ss]ource|src|SRC|project" -v

./pipeline.sh /home/jonas/Dokumente/TU_Darmstadt/thesis_orga/SEML/data/examples/test/ examples_out my_dataset2
