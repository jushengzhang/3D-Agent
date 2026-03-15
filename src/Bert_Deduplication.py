import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
# Initialize VLM annotation module
from VLM_preliminary_annotation_1 import VLMPreliminaryAnnotation
from transformers import RobertaModel, RobertaTokenizer
# import hdbscan  # Comment out hdbscan import
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from PIL import Image
import os
import json
import numpy as np

class BertDeduplicator(nn.Module):
    def __init__(self, model_name='FacebookAI/roberta-large', min_cluster_size=1, min_samples=1):
        super().__init__()
        # First download requires logging into huggingface
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name, use_auth_token=True)
        self.bert = RobertaModel.from_pretrained(model_name, use_auth_token=True)
        # Use DBSCAN with cosine distance
        self.clusterer = DBSCAN(  # Use sklearn.cluster.DBSCAN
            eps=0.3,          # Similar to HDBSCAN's cluster_selection_epsilon, needs adjustment based on data
            min_samples=min_samples,
            metric='cosine',
        )
        
        # Freeze BERT parameters (optional)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, queries, responses, scores):
        """
        Input:
            queries: List[str] list of queries
            responses: List[List[str]] list of responses for each query
            scores: List[List[float]] scores for each response
        
        Returns:
            List[Dict] deduplication results for each query:
                {
                    'canonical_response': str,
                    'max_score': float,
                    'cluster_members': List[str]
                }
        """
        results = []
        
        # Iterate through each query
        for query, res_list, score_list in zip(queries, responses, scores):
            # Generate BERT embeddings
            embeddings = self._get_embeddings(res_list)
            
            # Calculate cosine similarity matrix
            cos_sim = cosine_similarity(embeddings)
            
            # Use DBSCAN for clustering (input distance matrix)
            distance_matrix = 1 - cos_sim  # Convert similarity to distance
            clusters = self.clusterer.fit(distance_matrix).labels_ # DBSCAN uses fit().labels_
            
            # Add diagnostic output after clustering
            print(f"Number of noise points: {sum(clusters == -1)}")
            print(f"Number of clusters generated: {max(clusters)+1 if clusters.any() else 0}")
            
            # Filter noise points and merge results
            clustered_results = self._merge_clusters(res_list, score_list, clusters)
            
            results.append({
                'query': query,
                'deduplicated': clustered_results
            })
            
        return results

    def _get_embeddings(self, texts):
        """Generate normalized BERT embeddings"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
        
        # Use normalized pooled vectors
        embeddings = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)
        return embeddings.numpy()

    def _merge_clusters(self, responses, scores, clusters):
        """Improved cluster merging logic"""
        cluster_dict = {}
        
        for idx, (res, score) in enumerate(zip(responses, scores)):
            cluster_id = clusters[idx]
            
            # Skip noise points (-1)
            if cluster_id == -1:
                continue
                
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = {
                    'members': [],
                    'scores': []
                }
            
            cluster_dict[cluster_id]['members'].append(res)
            cluster_dict[cluster_id]['scores'].append(score)
        
        # Choose canonical response and highest score for each cluster
        final_results = []
        for cluster_id, data in cluster_dict.items():
            print(f"Scores for cluster {cluster_id}: {data['scores']}") # Print score list
            # Check if score list is empty
            if not data['scores']:
                print(f"Warning: Score list for cluster {cluster_id} is empty, skipping.")
                continue
            # Try to convert scores to float and handle potential errors
            try:
                numeric_scores = [float(s) for s in data['scores']] # Try to convert to float
                max_score = max(numeric_scores)
            except ValueError as e:
                print(f"Error: Score list for cluster {cluster_id} contains non-numeric elements: {data['scores']}")
                print(f"Error details: {e}")
                max_score = 0.0 # Or choose a default value, or raise an exception, based on your needs
            # Choose the highest scoring response as the canonical response
            canonical_idx = numeric_scores.index(max_score) # Use the converted numeric score list
            canonical_res = data['members'][canonical_idx]
            
            final_results.append({
                'canonical_response': canonical_res,
                'max_score': max_score,
                'cluster_members': data['members'],
                'cluster_id': int(cluster_id)
            })
            
        # Sort cluster results by highest score
        final_results = sorted(final_results, key=lambda x: x['max_score'], reverse=True)
        return final_results

class ClipWeightedDeduplicator(BertDeduplicator):
    def __init__(self, clip_model_name='openai/clip-vit-large-patch14',clip_weight_ratio=float(0.2), **kwargs):
        super().__init__(**kwargs)
        from transformers import CLIPModel, CLIPProcessor
        
        # Initialize CLIP model
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_weight_ratio = clip_weight_ratio
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, queries, responses, scores, images):
        """
        Additional parameter:
            images: List[PIL.Image] list of images for each query
        """
        # First perform BERT clustering deduplication
        base_results = super().forward(queries, responses, scores)
        
        # Calculate CLIP similarity weights
        for idx, (query_data, image) in enumerate(zip(base_results, images)):
            # Extract canonical response texts
            canonical_responses = [res['canonical_response'] for res in query_data['deduplicated']]
            
            # Calculate CLIP similarity
            clip_scores = self._get_clip_similarity(image, canonical_responses)
            
            # Output CLIP scores for each view
            view_name = os.path.splitext(os.path.basename(query_data['query']))[0]
            print(f"\nCLIP scores for view {view_name}:")
            for i, (response, score) in enumerate(zip(canonical_responses, clip_scores)):
                print(f"  Response {i+1}: {score:.4f}")
                print(f"  Text: {response[:50]}..." if len(response) > 50 else f"  Text: {response}")
                print("  ---")
            
            # Normalize to weights
            weights = F.softmax(torch.tensor(clip_scores), dim=0).tolist()
            
            # Update result data
            for res, weight in zip(query_data['deduplicated'], weights):
                print(res['max_score'])
                print(res['max_score'])
                res['clip_weight'] = float(weight)
                print(res['clip_weight'])
                res["Confidence"] = res['max_score']
                res['weighted_score'] = res["Confidence"] * (1 - self.clip_weight_ratio) + res['clip_weight'] * self.clip_weight_ratio

        # For each query result, keep only the cluster with the highest weighted score
        for query_data in base_results:
            if query_data['deduplicated']:  # Ensure there are deduplication results
                # Find the cluster with the highest weighted score
                max_weighted_cluster = max(query_data['deduplicated'], key=lambda x: x['weighted_score'])
                # Keep only this cluster
                query_data['deduplicated'] = [max_weighted_cluster]
        return base_results

    def _get_clip_similarity(self, image, texts):
        """Calculate CLIP similarity between image and text"""
        # Image processing
        image_inputs = self.clip_processor(
            images=image, 
            return_tensors="pt",
            padding=True
        ).to(self.clip_model.device)
        
        # Text processing
        text_inputs = self.clip_processor(
            text=texts,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=77  # CLIP text maximum length
        ).to(self.clip_model.device)
        
        # Feature extraction
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**image_inputs)
            text_features = self.clip_model.get_text_features(**text_inputs)
        
        # Normalize and calculate similarity
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        return (image_features @ text_features.T).squeeze(0).cpu().tolist()

class ResponseAggregator:
    def __init__(self, temperature=1.5):
        """
        Initialize aggregator
        Args:
            temperature (float): Temperature parameter for smoothing probability distribution
        """
        self.temperature = temperature

    def extract_first_sentence(self, text):
        """Extract the first sentence from text"""
        # Use common sentence ending separators to split
        for sep in ['. ', '! ', '? ']:
            if sep in text:
                return text.split(sep)[0] + sep.strip()
        return text  # If no separator found, return the entire text

    def aggregate_scores(self, canonical_responses, vlm_scores, clip_weights):
        """
        Aggregate multiple response scores
        Args:
            canonical_responses (list): List of canonical responses
            vlm_scores (list): List of corresponding VLM confidence scores
            clip_weights (list): List of corresponding CLIP weights
        Returns:
            dict: Dictionary containing final scores and probabilities
        """
        # Ensure inputs are numpy arrays and check dimensions
        vlm_scores = np.array(vlm_scores)
        clip_weights = np.array(clip_weights)
        
        # Check if input lengths are consistent
        if len(vlm_scores) != len(clip_weights) or len(vlm_scores) != len(canonical_responses):
            raise ValueError("Input array lengths are inconsistent")

        # Calculate weighted log-sum-exp
        weighted_scores = clip_weights * np.exp(vlm_scores)
        agg_scores = weighted_scores # Directly use weighted_scores

        # Ensure agg_scores is an array
        if np.isscalar(agg_scores):
            agg_scores = np.array([agg_scores])

        # Calculate initial softmax probabilities
        initial_probs = F.softmax(torch.tensor(agg_scores), dim=0).numpy()

        # Apply temperature scaling and sigmoid smoothing
        scaled_scores = agg_scores / self.temperature
        smoothed_probs = 1 / (1 + np.exp(-scaled_scores))

        # Normalize smoothed probabilities
        norm_probs = smoothed_probs / np.sum(smoothed_probs)

        # Return results
        results = []
        for i, res in enumerate(canonical_responses):
            results.append({
                'canonical_response': res,
                'aggregated_score': float(agg_scores[i]),
                'initial_probability': float(initial_probs[i]),
                'smoothed_probability': float(norm_probs[i])
            })

        # Sort by final probability
        results = sorted(results, key=lambda x: x['smoothed_probability'], reverse=True)
        
        return results

    def aggregate_multiple_views(self, view_results):
        """
        Aggregate results from multiple views, separately handling first sentence from front/back views and other content
        Args:
            view_results (list): List of aggregated results for each view
        Returns:
            dict: Dictionary containing final aggregated results
        """
        # Separate front/back views and other views
        front_back_responses = []
        front_back_scores = []
        front_back_weights = []
        
        other_responses = []
        other_scores = []
        other_weights = []

        for view in view_results:
            view_name = os.path.splitext(os.path.basename(view['query']))[0]
            
            if view_name in ['Front', 'Back']:
                # For front/back views, only process the first sentence
                for res in view['deduplicated']:
                    first_sentence = self.extract_first_sentence(res['canonical_response'])
                    front_back_responses.append(first_sentence)
                    front_back_scores.append(res['max_score'])
                    front_back_weights.append(res['clip_weight'])
            else:
                # Other views keep complete descriptions
                for res in view['deduplicated']:
                    other_responses.append(res['canonical_response'])
                    other_scores.append(res['max_score'])
                    other_weights.append(res['clip_weight'])

        # Separately aggregate front/back views and other views
        front_back_results = self.aggregate_scores(
            front_back_responses, 
            front_back_scores, 
            front_back_weights
        ) if front_back_responses else []

        other_results = self.aggregate_scores(
            other_responses, 
            other_scores, 
            other_weights
        ) if other_responses else []

        # Combine results: choose highest scoring front/back view description and other view description
        if front_back_results and other_results:
            best_front_back = front_back_results[0]['canonical_response']
            best_other = other_results[0]['canonical_response']
            
            # Combine descriptions
            combined_response = best_front_back + " " + best_other
            
            # Calculate combined score and probability
            combined_score = (front_back_results[0]['aggregated_score'] + 
                            other_results[0]['aggregated_score']) / 2
            combined_prob = (front_back_results[0]['smoothed_probability'] + 
                           other_results[0]['smoothed_probability']) / 2
            
            return [{
                'canonical_response': combined_response,
                'aggregated_score': combined_score,
                'smoothed_probability': combined_prob,
                'front_back_part': best_front_back,
                'other_part': best_other
            }]
        
        # If one part is empty, return results from the other part
        return front_back_results or other_results

class MABResponseAggregator(ResponseAggregator):
    def __init__(self, alpha=0.1, exploration_weight=1.0, learning_rate=0.1):
        """
        Multi-Armed Bandit based response aggregator
        
        Args:
            alpha (float): Initial prior parameter
            exploration_weight (float): UCB exploration weight
            learning_rate (float): Learning rate
        """
        super().__init__()
        self.alpha = alpha  # Prior parameter for reward distribution
        self.exploration_weight = exploration_weight
        self.learning_rate = learning_rate
        
        # Response history record
        self.response_history = {}  # {response_hash: {counts: count, reward_sum: total reward}}
    
    def calculate_ucb_scores(self, responses, initial_scores):
        """
        Calculate UCB scores for each response
        
        Args:
            responses (list): List of candidate responses
            initial_scores (list): List of initial scores
        
        Returns:
            list: List of UCB scores
        """
        total_pulls = sum(self.response_history.get(self._hash(r), {}).get('counts', 0) + self.alpha 
                          for r in responses)
        
        ucb_scores = []
        for i, response in enumerate(responses):
            response_hash = self._hash(response)
            
            # Get historical statistics for this response
            stats = self.response_history.get(response_hash, {'counts': 0, 'reward_sum': 0.0})
            counts = stats['counts'] + self.alpha  # Add prior
            
            # Calculate average reward
            if counts > 0:
                avg_reward = stats['reward_sum'] / counts
            else:
                avg_reward = initial_scores[i]  # Use initial score
            
            # Calculate UCB score
            exploration_term = self.exploration_weight * np.sqrt(2 * np.log(total_pulls) / counts)
            ucb_score = avg_reward + exploration_term
            
            ucb_scores.append(ucb_score)
        
        return ucb_scores
    
    def aggregate_scores(self, canonical_responses, vlm_scores, clip_weights):
        """
        Aggregate response scores using UCB strategy
        """
        # First use parent class method to calculate base aggregation results
        base_results = super().aggregate_scores(canonical_responses, vlm_scores, clip_weights)
        
        # Extract responses and initial scores
        responses = [r['canonical_response'] for r in base_results]
        initial_scores = [r['aggregated_score'] for r in base_results]
        
        # Calculate UCB scores
        ucb_scores = self.calculate_ucb_scores(responses, initial_scores)
        
        # Update scores in results
        for i, res in enumerate(base_results):
            res['ucb_score'] = float(ucb_scores[i])
        
        # Sort by UCB score
        results = sorted(base_results, key=lambda x: x['ucb_score'], reverse=True)
        
        return results
    
    def update_reward(self, response, reward):
        """
        Update reward history for a response
        
        Args:
            response (str): Response text
            reward (float): Reward received
        """
        response_hash = self._hash(response)
        
        if response_hash not in self.response_history:
            self.response_history[response_hash] = {
                'counts': 0,
                'reward_sum': 0.0
            }
        
        # Update statistics
        self.response_history[response_hash]['counts'] += 1
        self.response_history[response_hash]['reward_sum'] += reward
    
    def _hash(self, text):
        """Generate hash value for a response"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def save_history(self, filepath):
        """Save learning history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.response_history, f)
    
    def load_history(self, filepath):
        """Load learning history from file"""
        try:
            with open(filepath, 'r') as f:
                self.response_history = json.load(f)
        except FileNotFoundError:
            print(f"History file {filepath} does not exist, using empty history.")

# Simulated user feedback collector
class UserFeedbackCollector:
    def __init__(self, mab_aggregator):
        self.mab_aggregator = mab_aggregator
    
    def collect_feedback(self, response, feedback_score):
        """
        Collect user feedback and update MAB model
        
        Args:
            response (str): Response shown to user
            feedback_score (float): Rating provided by user (0-1)
        """
        self.mab_aggregator.update_reward(response, feedback_score)
        print(f"Feedback collected and model updated: '{response[:30]}...' - Rating: {feedback_score}")

if __name__ == "__main__":
    
    # Initialize modules
    vlm_annotator = VLMPreliminaryAnnotation()
    deduplicator = ClipWeightedDeduplicator()
    # Use MAB aggregator instead of regular aggregator
    aggregator = MABResponseAggregator(alpha=0.1, exploration_weight=0.5)
    
    # Try to load historical data
    try:
        aggregator.load_history("mab_history.json")
        print("Successfully loaded MAB historical data")
    except:
        print("No historical data found, will use new model")
    
    # Create user feedback collector
    feedback_collector = UserFeedbackCollector(aggregator)
    
    # Example image paths
    test_images = [
        "../tmp/Front.png",
        "../tmp/Back.png", 
        "../tmp/Left.png",
        "../tmp/Right.png",
        "../tmp/Up.png",
        "../tmp/Down.png"
    ]

    print(test_images)
    
    # Step 1: Get preliminary annotations through VLM
    print("Generating preliminary annotations through VLM...")
    # If annotation_results.json already exists, read directly
    if os.path.exists("annotation_results.json"):
        print("Reading annotation_results.json")
        with open("annotation_results.json", "r") as f:
            vlm_results = json.load(f)
    else:
        print("Generating annotation_results.json")
        vlm_results = vlm_annotator(test_images)

    
    # Convert data format
    queries = []
    responses = []
    scores = []
    images = []
    
    # Iterate through keys (image paths) in the results dictionary
    for image_path in vlm_results['results'].keys():
        # Load corresponding image
        img = Image.open(image_path)

        # Extract all candidate responses for this image path
        view_data = vlm_results['results'][image_path]
        candidate_list = view_data['candidates'] # Get candidate list
        res_list = [candidate['text'] for candidate in candidate_list]  # Response texts
        score_list = [candidate['confidence_score'] for candidate in candidate_list]  # Confidence scores

        queries.append(image_path) # Use image path as query
        responses.append(res_list)
        scores.append(score_list)
        images.append(img)
    
    # Step 2: Perform deduplication and weighting
    print("\nPerforming deduplication and weighting...")
    final_results = deduplicator(queries, responses, scores, images)
    
    # Step 3: Aggregate results from all views
    print("\nPerforming global aggregation...")
    global_results = aggregator.aggregate_multiple_views(final_results)

    # Step 4: Print global aggregation results
    print("\nGlobal aggregation results:")
    for i, res in enumerate(global_results, 1):
        print(f"Rank {i}:")
        print(f"  Canonical response: {res['canonical_response']}")
        print(f"  Aggregated score: {res['aggregated_score']:.4f}")
        print(f"  Smoothed probability: {res['smoothed_probability']:.4f}")
        if 'ucb_score' in res:
            print(f"  UCB score: {res['ucb_score']:.4f}")
        print("---")
    
    # Step 5: Simulate collecting user feedback (in a real system, this would be collected through UI)
    if len(global_results) > 0:
        print("\nSimulating user feedback (in a real application, this would involve UI interaction):")
        top_response = global_results[0]['canonical_response']
        mock_feedback = 0.9  # Simulate high rating, in reality would be provided by user
        feedback_collector.collect_feedback(top_response, mock_feedback)
    
    # Save updated MAB history
    aggregator.save_history("mab_history.json")
    print("MAB history saved")
