from typing import List
import csv
import torch
import sys

class ProteinTokenizer:
    def __init__(self):
        """Initialize tokenizer with special tokens and amino acid vocabulary"""
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'  # X for unknown amino acids
        self.pad_token = '<PAD>'
        self.vocab = {aa: idx + 1 for idx, aa in enumerate(self.amino_acids)}  # Start from 1
        self.vocab[self.pad_token] = 0  # PAD token gets 0
        
    def tokenize(self, sequences: List[str], max_length: int = None) -> torch.Tensor:
        """Convert sequences to tensor, padding if necessary"""
        batch_size = len(sequences)
        tensor = torch.zeros((batch_size, max_length), dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:max_length]):
                # Use 0 for unknown amino acids
                tensor[i, j] = self.vocab.get(aa, self.vocab['X'])
                
        return tensor

def read_m8_file(filename: str) -> List[str]:
    """Read protein sequences from M8 file"""
    sequences = []
    
    with open(filename, 'r') as f:
        # Skip header
        next(f)
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            # Get sequence from third column
            sequences.append(row[2])
    
    return sequences

def main():
    # Check if filepath is provided
    if len(sys.argv) != 2:
        print("Usage: python protein_stuff.py <path_to_m8_file>")
        sys.exit(1)
        
    # Get filepath from command line argument
    filepath = sys.argv[1]
    
    # Read sequences
    sequences = read_m8_file(filepath)
    
    # Create tokenizer and convert sequences
    tokenizer = ProteinTokenizer()
    tensor = tokenizer.tokenize(sequences)
    
    # Print information about the tensor
    print(f"Tensor shape: {tensor.shape}")
    print("\nFirst sequence tokens:")
    print(tensor[0][:20])  # First 20 tokens of first sequence
    print(tensor)
    
    # Print total number of sequences
    print(f"\nTotal sequences: {len(sequences)}")

if __name__ == "__main__":
    main()