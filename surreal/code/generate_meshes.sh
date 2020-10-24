# Loop Over Body Shapes
for R in 519 1320 521 523 779 365 1198 368 337 944 1333 502 344 538 413
do
	# Loop over Genders
	for GENDER in male female
	do
		# Walking humans
		for i in 2 5 7 8 13 14 24 26 27 46 48 49 59 60 64 65
		do
			$BLENDER_PATH/blender -b -t 1 -P export_human_meshes.py -- --idx $i --ishape 0 --stride 50 --gender $GENDER --body_shape_idx $R --outdir human_meshes
		done

		# Stationary humans
		for i in 132 133 134 135 
		do
			$BLENDER_PATH/blender -b -t 1 -P export_human_meshes.py -- --idx $i --ishape 0 --stride 50 --gender $GENDER --body_shape_idx $R --outdir human_meshes
		done

	done
done

# Make sure the generated meshes match what is expected
$BLENDER_PATH/blender -b -P verify_human_mesh_generation.py

# Collect the relevant textures needed for the meshes
$BLENDER_PATH/blender -b -P collect_human_textures.py
