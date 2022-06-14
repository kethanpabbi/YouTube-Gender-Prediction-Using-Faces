from pixellib.semantic import semantic_segmentation

segment_video = semantic_segmentation()
segment_video.load_pascalvoc_model("/Users/kethanpabbi/Downloads/Gender-Detection-master/Segmentation/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
segment_video.process_video_pascalvoc("/Users/kethanpabbi/Downloads/Gender-Detection-master/Segmentation/Pexels Videos 2796078.mp4",  overlay = True, frames_per_second= 15, output_video_name="path_to_output_video")
