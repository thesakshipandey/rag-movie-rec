#!/usr/bin/env python3
"""Analyze diversity and quality of generated pairs for prompts 0001-0010"""
import json
import os

def analyze_prompt_results(prompt_id):
    """Analyze results for a single prompt"""
    json_path = f"results/{prompt_id}.json"
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    pairs = data.get('pairs', [])
    if not pairs:
        return None
    
    # Count movie appearances
    movie_count = {}
    for pair in pairs:
        m1_id = pair.get('movie1_id')
        m2_id = pair.get('movie2_id')
        movie_count[m1_id] = movie_count.get(m1_id, 0) + 1
        movie_count[m2_id] = movie_count.get(m2_id, 0) + 1
    
    unique_movies = len(movie_count)
    diversity_score = (unique_movies / 18) * 100
    
    # Find repeated movies
    repeated = [(mid, count) for mid, count in movie_count.items() if count > 1]
    
    return {
        'prompt_id': prompt_id,
        'total_pairs': len(pairs),
        'unique_movies': unique_movies,
        'diversity_score': diversity_score,
        'repeated_movies': len(repeated),
        'max_appearances': max(movie_count.values()) if movie_count else 0,
        'pairs': pairs
    }

def main():
    print("=" * 80)
    print("MOVIE PAIR DATASET ANALYSIS: Prompts 0001-0010")
    print("=" * 80)
    print()
    
    results = []
    for i in range(1, 11):
        prompt_id = f"{i:04d}"
        analysis = analyze_prompt_results(prompt_id)
        if analysis:
            results.append(analysis)
    
    if not results:
        print("❌ No results found. Run the generator first.")
        return
    
    print(f"✓ Analyzed {len(results)} prompts\n")
    
    # Summary table
    print("┌" + "─" * 78 + "┐")
    print(f"│{'Prompt':^10}│{'Pairs':^8}│{'Unique':^10}│{'Diversity':^12}│{'Repeated':^10}│{'Max Apps':^10}│")
    print("├" + "─" * 78 + "┤")
    
    total_diversity = 0
    perfect_count = 0
    
    for r in results:
        diversity_indicator = "✓" if r['diversity_score'] == 100 else "⚠"
        print(f"│{r['prompt_id']:^10}│{r['total_pairs']:^8}│{r['unique_movies']:^10}│"
              f"{r['diversity_score']:>10.1f}% {diversity_indicator}│{r['repeated_movies']:^10}│{r['max_appearances']:^10}│")
        
        total_diversity += r['diversity_score']
        if r['diversity_score'] == 100:
            perfect_count += 1
    
    print("└" + "─" * 78 + "┘")
    print()
    
    # Overall stats
    avg_diversity = total_diversity / len(results)
    print(f"📊 Overall Statistics:")
    print(f"  Average Diversity: {avg_diversity:.1f}%")
    print(f"  Perfect Diversity (100%): {perfect_count}/{len(results)} prompts")
    print(f"  Success Rate: {(perfect_count/len(results)*100):.1f}%")
    print()
    
    # Show problem cases
    problem_prompts = [r for r in results if r['diversity_score'] < 100]
    if problem_prompts:
        print(f"⚠️  Problem Prompts (< 100% diversity):")
        for r in problem_prompts:
            print(f"\n  Prompt {r['prompt_id']} ({r['diversity_score']:.1f}% diversity):")
            
            # Find which movies are repeated
            movie_count = {}
            for pair in r['pairs']:
                m1_id, m2_id = pair['movie1_id'], pair['movie2_id']
                movie_count[m1_id] = movie_count.get(m1_id, 0) + 1
                movie_count[m2_id] = movie_count.get(m2_id, 0) + 1
            
            repeated = [(mid, count) for mid, count in sorted(movie_count.items(), key=lambda x: -x[1]) if count > 1]
            
            for movie_id, count in repeated[:3]:  # Show top 3 repeated
                pairs_with_movie = [p for p in r['pairs'] if p['movie1_id'] == movie_id or p['movie2_id'] == movie_id]
                title = pairs_with_movie[0]['movie1_title'] if pairs_with_movie[0]['movie1_id'] == movie_id else pairs_with_movie[0]['movie2_title']
                print(f"    [{movie_id}] {title}: {count}× appearances")
    else:
        print("🎉 Perfect! All prompts achieved 100% diversity!")
    
    print()

if __name__ == "__main__":
    main()
