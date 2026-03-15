
import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Add Uni3D path
UNI3D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Uni3D')
sys.path.append(UNI3D_PATH)

# Import Uni3D related modules
from Uni3D.models.PointCloudTextSimilarityGating import (
    create_point_cloud_text_gating, 
    create_uni3d_gating_deduplicator,
    PointCloudTextSimilarityGating,
    Uni3DGatingDeduplicator
)
from Uni3D.models.uni3d import create_uni3d
from models.Bert_Deduplication import ResponseAggregator, MABResponseAggregator, UserFeedbackCollector


class PointCloudGating:
    """
    Point Cloud Gating Interface Class
    
    Provides a simplified interface for calling point cloud-text similarity gating functionality,
    used to filter descriptions inconsistent with the geometric features of the point cloud.
    """
    
    def __init__(self, 
                 pc_model='vit_base_patch16_224',
                 pretrained_pc=None,
                 threshold=0.577,
                 use_deduplication=True,
                 clip_model_name='openai/clip-vit-large-patch14',
                 clip_weight_ratio=0.2,
                 bert_model_name='FacebookAI/roberta-large',
                 device=None):
        """
        Initialize the Point Cloud Gating Interface
        
        Parameters:
            pc_model (str): Point cloud encoder model name
            pretrained_pc (str): Path to the pretrained point cloud model
            threshold (float): Similarity threshold, default is 0.577
            use_deduplication (bool): Whether to use deduplication functionality
            clip_model_name (str): CLIP model name
            clip_weight_ratio (float): CLIP weight ratio
            bert_model_name (str): BERT model name
            device (str): Device type, such as 'cuda' or 'cpu'
        """
        self.threshold = threshold
        self.use_deduplication = use_deduplication
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create arguments object
        class Args:
            def __init__(self):
                self.pc_model = pc_model
                self.pretrained_pc = pretrained_pc
                self.drop_path_rate = 0.0
        
        self.args = Args()
        
        # Initialize Uni3D model
        print(f"Initializing Uni3D model (using device: {self.device})...")
        self.uni3d_model = create_uni3d(self.args)
        self.uni3d_model = self.uni3d_model.to(self.device)
        self.uni3d_model.eval()
        
        # Initialize gating proxy or deduplicator based on configuration
        if use_deduplication:
            print("Initializing point cloud-text similarity gating deduplicator...")
            self.gating_agent = create_uni3d_gating_deduplicator(
                args=self.args,
                uni3d_model=self.uni3d_model,
                similarity_threshold=threshold,
                clip_model_name=clip_model_name,
                clip_weight_ratio=clip_weight_ratio,
                bert_model_name=bert_model_name
            )
        else:
            print("Initializing point cloud-text similarity gating proxy...")
            self.gating_agent = create_point_cloud_text_gating(
                args=self.args,
                uni3d_model=self.uni3d_model,
                threshold=threshold
            )
        
        # Create aggregator
        self.aggregator = ResponseAggregator()
    
    def load_point_cloud(self, pc_path):
        """
        Load point cloud data
        
        Parameters:
            pc_path (str): Path to the point cloud file
            
        Returns:
            torch.Tensor: Point cloud data, shape [1, num_points, 6]
        """
        try:
            import trimesh
            mesh = trimesh.load(pc_path)
            
            # Extract point cloud coordinates and colors
            vertices = np.array(mesh.vertices)
            
            # Check if color information exists
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                colors = np.array(mesh.visual.vertex_colors[:, :3]) / 255.0
            else:
                # If no color information, use zero vectors
                colors = np.zeros((vertices.shape[0], 3))
            
            # Combine coordinates and colors
            pc = np.concatenate([vertices, colors], axis=1)
            pc = torch.from_numpy(pc).float().unsqueeze(0)  # Add batch dimension
            return pc.to(self.device)
        except Exception as e:
            print(f"Warning: Error loading point cloud file {pc_path}: {e}")
            # Return default point cloud (empty point cloud)
            return torch.zeros((1, 1024, 6)).to(self.device)
    
    def filter_responses(self, 
                        point_cloud_path, 
                        responses, 
                        scores=None, 
                        image_path=None, 
                        critical_categories=None):
        """
        Filter text responses using point cloud similarity
        
        Parameters:
            point_cloud_path (str): Path to the point cloud file
            responses (list): List of text responses
            scores (list, optional): List of confidence scores for the responses
            image_path (str, optional): Corresponding image path for deduplication mode
            critical_categories (list, optional): List of key categories
            
        Returns:
            dict: Filtered/deduplicated results
        """
        # Load point cloud data
        point_cloud = self.load_point_cloud(point_cloud_path)
        
        # Use different filtering methods based on whether deduplication is enabled
        if self.use_deduplication:
            if image_path is None:
                image_path = point_cloud_path  # If no image path is provided, use the point cloud path as an identifier
            
            # Load image (if path is provided)
            image = None
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    print(f"Warning: Unable to load image {image_path}: {e}")
            
            # Prepare input data
            queries = [image_path]
            responses_list = [responses]
            scores_list = [scores if scores is not None else [1.0] * len(responses)]
            images = [image]
            point_clouds = [point_cloud]
            
            # Apply deduplication and gating
            results = self.gating_agent(
                queries, 
                responses_list, 
                scores_list, 
                images, 
                point_clouds, 
                critical_categories
            )
            
            # Aggregate results (optional)
            if len(results) > 0:
                global_results = self.aggregator.aggregate_multiple_views(results)
                return {
                    'per_view_results': results,
                    'global_results': global_results
                }
            return {'per_view_results': results}
        else:
            # Directly use point cloud-text similarity gating
            if not isinstance(responses, list):
                responses = [responses]
            
            if scores is not None and not isinstance(scores, list):
                scores = [scores]
            
            # Apply gating
            result = self.gating_agent(
                point_cloud, 
                responses, 
                scores, 
                critical_categories
            )
            
            return result

    def process_vlm_results(self, 
                           vlm_results, 
                           point_cloud_dir, 
                           critical_categories=None):
        """
        Process VLM results, applying point cloud gating to each image path
        
        Parameters:
            vlm_results (dict): VLM results dictionary
            point_cloud_dir (str): Directory of point cloud files
            critical_categories (list, optional): List of key categories
            
        Returns:
            dict: Processed results
        """
        queries = []
        responses = []
        scores = []
        images = []
        point_clouds = []
        
        # Process each image path
        print("\nPreparing data...")
        for image_path in vlm_results['results'].keys():
            # Load image
            img = None
            try:
                img = Image.open(image_path)
            except Exception as e:
                print(f"Warning: Unable to load image {image_path}: {e}")
            
            # Load point cloud data
            pc_filename = os.path.basename(image_path).split('.')[0] + '.ply'
            pc_path = os.path.join(point_cloud_dir, pc_filename)
            pc = self.load_point_cloud(pc_path)
            
            # Get candidate responses
            view_data = vlm_results['results'][image_path]
            candidate_list = view_data['candidates']
            res_list = [candidate['text'] for candidate in candidate_list]
            score_list = [candidate['confidence_score'] for candidate in candidate_list]
            
            # Add to lists
            queries.append(image_path)
            responses.append(res_list)
            scores.append(score_list)
            images.append(img)
            point_clouds.append(pc)
        
        # Choose different processing methods based on whether deduplication is used
        if self.use_deduplication:
            # Use full gating deduplication process
            print("\nUsing point cloud-text similarity gating deduplicator...")
            final_results = self.gating_agent(
                queries, 
                responses, 
                scores, 
                images, 
                point_clouds, 
                critical_categories
            )
        else:
            # Only use point cloud-text similarity gating
            print("\nUsing only point cloud-text similarity gating...")
            final_results = []
            for i, (query, pc, res_list, score_list) in enumerate(zip(queries, point_clouds, responses, scores)):
                # Apply point cloud-text similarity gating
                gating_result = self.gating_agent(
                    pc, 
                    res_list, 
                    score_list, 
                    critical_categories
                )
                
                # Get filtered responses and scores
                filtered_responses = gating_result['filtered_responses']
                filtered_scores = gating_result.get('filtered_scores', [])
                manual_review_flags = gating_result['manual_review_flags']
                
                # Construct results
                canonical_results = []
                for j, (resp, score, flag) in enumerate(zip(filtered_responses, filtered_scores, manual_review_flags)):
                    canonical_results.append({
                        'canonical_response': resp,
                        'max_score': score,
                        'needs_manual_review': flag,
                        'cluster_members': [resp]
                    })
                
                final_results.append({
                    'query': query,
                    'deduplicated': canonical_results
                })
        
        # Aggregate results from all views
        print("\nPerforming global aggregation...")
        global_results = self.aggregator.aggregate_multiple_views(final_results)
        
        return {
            'per_view_results': final_results,
            'global_results': global_results
        }


# Example usage
if __name__ == "__main__":
    import argparse
    import json
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Point Cloud-Text Similarity Gating Example')
    parser.add_argument('--vlm_results', type=str, required=True, help='VLM results JSON file path')
    parser.add_argument('--point_cloud_dir', type=str, required=True, help='Point cloud file directory')
    parser.add_argument('--pretrained_pc', type=str, required=True, help='Pretrained point cloud model path')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--similarity_threshold', type=float, default=0.577, help='Similarity threshold')
    parser.add_argument('--only_gating', action='store_true', help='Whether to only use gating functionality, without deduplication')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize point cloud gating
    gating = PointCloudGating(
        pretrained_pc=args.pretrained_pc,
        threshold=args.similarity_threshold,
        use_deduplication=not args.only_gating
    )
    
    # Load VLM results
    with open(args.vlm_results, 'r') as f:
        vlm_results = json.load(f)
    
    # Process VLM results
    results = gating.process_vlm_results(
        vlm_results, 
        args.point_cloud_dir,
        critical_categories=['chair', 'table', 'desk', 'sofa']  # Example critical categories
    )
    
    # Save results
    output_path = os.path.join(args.output_dir, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print global aggregation results
    print("\nGlobal aggregation results:")
    for i, res in enumerate(results['global_results'], 1):
        print(f"Rank {i}:")
        print(f"   Canonical response: {res['canonical_response']}")
        print(f"   Aggregated score: {res['aggregated_score']:.4f}")
        print(f"   Smoothed probability: {res['smoothed_probability']:.4f}")
        if 'needs_manual_review' in res and res['needs_manual_review']:
            print(f"   Needs manual review: Yes")
        print("---")
