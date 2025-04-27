import cityhash
from baseline.aggregated_features_baseline import create_embeddings
import torch
from pathlib import Path
import sys, os


def get_bin(ids, seed=42, n_bins=512):
  transform = lambda x: cityhash.CityHash64(f"p/{seed}/{x}") % n_bins
  return torch.tensor(list(map(transform, ids)), dtype=int)

def one_hot_embed(bins, n_bins=512):
  embeds = torch.zeros((bins.size(0), n_bins))
  embeds.scatter_add_(1, bins.unsqueeze(1), torch.ones(bins.size(0), 1))
  return embeds

def create_multihash_embeddings(ids, seeds=[42, 666, 777, 4], n_bins=10):
  bins = [get_bin(ids, seed, n_bins) for seed in seeds]
  result = []
  for bin in bins:
    result += [one_hot_embed(bin, n_bins)] 
  return ids, torch.cat(result)

if __name__ == "__main__":

    relevant_ids = create_embeddings.load_relevant_clients_ids(Path("/content/dataset/input"))
    client_ids, embeddings = create_multihash_embeddings(relevant_ids)
    embeddings_dir = Path("/content/dataset/output")


    create_embeddings.save_embeddings(
        client_ids=client_ids,
        embeddings=embeddings,
        embeddings_dir=embeddings_dir,
    )