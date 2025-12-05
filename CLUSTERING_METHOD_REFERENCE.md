# Clustering Method Reference

## Method: Parsimony-Adjusted Silhouette Score Selection

### Approach
We use a threshold-based method to select the optimal number of clusters (k) that balances:
1. **Quality**: High silhouette score (good cluster separation and cohesion)
2. **Parsimony**: Fewer clusters (simpler model)

### Implementation
1. Calculate silhouette scores for k from 2 to 40
2. Find the maximum silhouette score
3. Set threshold = 95% of maximum silhouette score
4. Select the **smallest k** where silhouette score ≥ threshold

This ensures we get high-quality clustering (within 5% of best) while minimizing the number of clusters (parsimony principle).

### Rationale
This method is inspired by:
- **Elbow Method**: Identifies the point of diminishing returns
- **Parsimony Principle**: Prefer simpler models when quality is similar (Occam's Razor)
- **Silhouette Analysis**: Uses Rousseeuw's silhouette coefficient for validation

### Academic References

1. **Silhouette Coefficient (Original Method)**
   - Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis". Journal of Computational and Applied Mathematics, 20, 53-65.
   - DOI: 10.1016/0377-0427(87)90125-7

2. **Elbow Method**
   - Wikipedia: "Elbow method (clustering)" - https://en.wikipedia.org/wiki/Elbow_method_(clustering)
   - Identifies optimal k by finding the "elbow" where adding more clusters provides diminishing returns

3. **Parsimony Principle in Clustering**
   - The principle of parsimony (Occam's Razor) suggests choosing simpler models when multiple models explain the data equally well
   - Applied here: if multiple k values achieve similar silhouette scores (within 5%), choose the smallest k

### Why 95% Threshold?
- 95% threshold balances quality and parsimony
- Allows for slight quality trade-off to achieve significantly fewer clusters
- Similar to elbow method's concept of "diminishing returns"
- Can be adjusted (e.g., 90% for more parsimony, 98% for higher quality)
