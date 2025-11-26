""" Baseline of solving V-WSD with CLIP """
import argparse
import logging
import os
from os.path import join as pj

import pandas as pd
from vwsd import CLIP, MultilingualCLIP, data_loader, plot

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description="Solve V-WSD")
    parser.add_argument('-d', '--data-dir', help='directly of images', default='dataset', type=str)
    parser.add_argument('-l', '--language', help='language', default='en', type=str)
    parser.add_argument('-m', '--model-clip', help='clip model', default=None, type=str)
    parser.add_argument('-o', '--output-dir', help='output directly', default="result", type=str)
    parser.add_argument('-p', '--prompt', help='prompt to be used in text embedding (specify the placeholder by <>)',
                        type=str, nargs='+',
                        default=['<>', 'This is <>.', 'Example of an image caption that explains <>.'])
    parser.add_argument('--input-type', help='input text type',
                        type=str, nargs='+', default=['target word', 'target phrase'])
    parser.add_argument('-b', '--batch-size', help='batch size', default=None, type=int)
    parser.add_argument('--plot', help='', action='store_true')
    parser.add_argument('--use-image-query', help='use image-to-image similarity instead of text-to-image', action='store_true')
    parser.add_argument('-i', '--image-dir', help='directory containing query images for image-to-image mode', default='image', type=str)
    parser.add_argument('--image-pattern', help='naming pattern for query images (use {n} for sample number)', 
                        default='generated_{n}.jpg', type=str)
    parser.add_argument('--image-index-offset', help='offset for image index (e.g., 1 if images start from 1.jpg)', 
                        default=0, type=int)
    opt = parser.parse_args()

    # sanity check
    if not opt.use_image_query:
        assert all("<>" in p for p in opt.prompt), "prompt need to contain `<>`"
    # os.makedirs(opt.output_dir, exist_ok=True)

    # load dataset
    data = data_loader(opt.data_dir)[opt.language]

    # load model (only English CLIP supports image-to-image similarity)
    if opt.language == 'en':
        clip = CLIP(opt.model_clip if opt.model_clip is not None else 'openai/clip-vit-large-patch14-336')
    else:
        if opt.use_image_query:
            logging.warning("Image-to-image mode only supports English. Falling back to text-to-image mode.")
            opt.use_image_query = False
        clip = MultilingualCLIP(
            opt.model_clip if opt.model_clip is not None else 'sentence-transformers/clip-ViT-B-32-multilingual-v1')

    # run inference
    result = []
    total_samples = len(data)
    processed_count = 0
    skipped_count = 0
    
    logging.info(f"{'='*60}")
    logging.info(f"Starting inference on {total_samples} samples")
    logging.info(f"{'='*60}")
    
    for n, d in enumerate(data):
        progress_pct = (n + 1) / total_samples * 100
        logging.info(f"[{n+1}/{total_samples}] ({progress_pct:.1f}%) - {d['target phrase']}")
        
        if opt.use_image_query:
            # Image-to-image similarity mode
            # Apply index offset (e.g., if images start from 1.jpg instead of 0.jpg)
            image_index = n + opt.image_index_offset
            query_image_path = pj(opt.image_dir, opt.image_pattern.format(n=image_index))
            
            if not os.path.exists(query_image_path):
                logging.warning(f"  ⚠️  Query image not found: {query_image_path}, skipping")
                skipped_count += 1
                continue
            
            logging.info(f"  🖼️  Query: {os.path.basename(query_image_path)}")
            
            # Calculate image-to-image similarity
            logging.info(f"  🔄 Computing similarity with {len(d['candidate images'])} candidates...")
            sim = clip.get_image_similarity(query_image_path, d['candidate images'], batch_size=opt.batch_size)
            processed_count += 1
            logging.info(f"  ✅ Done! Best match: {os.path.basename(sorted(zip(sim[0], d['candidate images']), key=lambda x: x[0], reverse=True)[0][1])}")
            
            if opt.plot:
                # Save to a separate folder for image-to-image mode
                plot(
                    similarity=sim,
                    texts=[f"Query Image: {os.path.basename(query_image_path)}"],
                    images=d['candidate images'],
                    export_file=pj(opt.output_dir, "visualization", "image_to_image", opt.language, f'similarity.{n}.png')
                )
            
            # Only one query (the image), so sim[0] contains all similarities
            tmp = sorted(zip(sim[0], d['candidate images']), key=lambda x: x[0], reverse=True)
            result.append({
                'language': opt.language,
                'data': n,
                'candidate': [os.path.basename(i[1]) for i in tmp],
                'relevance': sorted(sim[0], reverse=True),
                'query_image': os.path.basename(query_image_path),
                'input_type': 'image_query',
                'prompt': 'image_to_image'
            })
        else:
            # Original text-to-image similarity mode
            prompt_list = []
            for input_type in opt.input_type:
                prompt_list += [(p.replace("<>", d[input_type]), input_type, p) for p in opt.prompt]

            sim = clip.get_similarity(texts=[p[0] for p in prompt_list], images=d['candidate images'], batch_size=opt.batch_size)
            if opt.plot:
                plot(
                    similarity=sim,
                    texts=[p[0] for p in prompt_list],
                    images=d['candidate images'],
                    export_file=pj(opt.output_dir, "visualization", opt.language, f'similarity.{n}.png')
                )
            for (text, input_type, prompt_type), s in zip(prompt_list, sim):
                tmp = sorted(zip(s, d['candidate images']), key=lambda x: x[0], reverse=True)
                result.append({
                    'language': opt.language,
                    'data': n,
                    'candidate': [os.path.basename(i[1]) for i in tmp],
                    'relevance': sorted(s, reverse=True),
                    'text': text,
                    'input_type': input_type,
                    'prompt': prompt_type
                })

    df = pd.DataFrame(result)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Inference completed!")
    logging.info(f"  Total samples: {total_samples}")
    logging.info(f"  Processed: {processed_count}")
    logging.info(f"  Skipped: {skipped_count}")
    logging.info(f"{'='*60}\n")
    
    if opt.use_image_query:
        # For image-to-image mode, use a simpler output structure
        path = pj(opt.output_dir, 'image_to_image_similarity')
        os.makedirs(path, exist_ok=True)
        with open(pj(path, f'prediction.{opt.language}.txt'), 'w') as f:
            f.write('\n'.join(['\t'.join(x) for x in df.sort_values(by=['data'])['candidate'].to_list()]))
        df.to_csv(pj(path, f'full_result.{opt.language}.csv'), index=False)
        logging.info(f"Results saved to {path}")
    else:
        # Original text-to-image mode output
        for (prompt, input_type), g in df.groupby(['prompt', 'input_type']):
            path = pj(opt.output_dir, f'{prompt.replace("<>", "mask")}.{input_type}'.replace(" ", "_"))
            os.makedirs(path, exist_ok=True)
            with open(pj(path, f'prediction.{opt.language}.txt'), 'w') as f:
                f.write('\n'.join(['\t'.join(x) for x in g.sort_values(by=['data'])['candidate'].to_list()]))
            g.to_csv(pj(path, f'full_result.{opt.language}.csv'), index=False)


if __name__ == '__main__':
    main()
