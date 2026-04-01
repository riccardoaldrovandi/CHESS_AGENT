import torch
import os
import chess
import numpy as np

# --- Le tue funzioni di salvataggio ---
def save_checkpoint(model, optimizer, iteration, path="checkpoints/chess_model.pth"):
    # Crea la cartella checkpoints se non esiste
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"--> Checkpoint salvato: iterazione {iteration}")

def load_checkpoint(model, optimizer, path="checkpoints/chess_model.pth"):
    if os.path.exists(path):
        # Fondamentale: carica sulla GPU o CPU correttamente
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"--> Ripristinato da iterazione {checkpoint['iteration']}")
        return checkpoint['iteration']
    return 0

def board_to_tensor(board):
    # Inizializziamo una matrice vuota 12x8x8 con zeri
    # 12 piani (6 pezzi x 2 colori), 8 righe, 8 colonne
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Mappa dei pezzi per python-chess
    # P=1, N=2, B=3, R=4, Q=5, K=6 (Bianchi)
    # neri sono gli stessi ma negativi per la libreria, noi li separiamo nei canali 6-11
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Calcoliamo riga e colonna (0-7)
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            
            # Determiniamo il piano (0-5 per bianchi, 6-11 per neri)
            channel = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            
            tensor[channel, row, col] = 1.0

    # Trasformiamo l'array numpy in un tensore PyTorch e aggiungiamo la dimensione "Batch"
    # La rete si aspetta [Batch_Size, Canali, Altezza, Larghezza] -> [1, 12, 8, 8]
    input_tensor = torch.from_numpy(tensor).unsqueeze(0)
    
    # Spediamo il tensore sulla GPU (se disponibile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return input_tensor.to(device)