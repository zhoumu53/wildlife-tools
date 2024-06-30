
### Demo -- get topk indices for query feature

import timm
from wildlife_tools.data import FeatureDatabase
from wildlife_tools.inference import KnnMatcher

model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)

feature_path = 'features.pkl'
database = FeatureDatabase.from_file(feature_path)
matcher = KnnMatcher(database)

metadata = database.metadata

n_query = 1
k = 10
query_features = database.features[:n_query]
gallery_features = database.features[n_query:]
query_metadata = metadata[:n_query]
gallery_metadata = metadata[n_query:]

col_path='path'
col_label='identity'

topk_paths, topk_labels = matcher.get_topk_indices(query_features, gallery_features, col_path=col_path, col_label=col_label, k=k)

for i, (query_label, query_path) in enumerate(zip(query_metadata[col_label], query_metadata[col_path] )):

    print(f'query: {query_label} ({query_path})')
    print(f'topk_labels: {topk_labels[i]}')
    print(f'topk_paths: {topk_paths[i]}')
    print()