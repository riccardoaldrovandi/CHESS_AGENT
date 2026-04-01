import torch
import torch.nn as nn
import torch.optim as optim
import os

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # --- BLOCCO CONVOLUZIONALE (Il "Cervello Visivo") ---
        # Analizza la disposizione spaziale dei pezzi sulla scacchiera 8x8.
        self.conv_block = nn.Sequential(
            # Primo strato: prende 12 piani (pezzi) e ne crea 64 mappe di caratteristiche.
            # kernel_size=3 guarda quadratini 3x3. padding=1 mantiene la dimensione 8x8.
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(), # Funzione di attivazione: rende la rete capace di imparare pattern complessi
            
            # Secondo strato: approfondisce l'analisi portando le mappe da 64 a 128.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # --- STRATO DI APPIATTIMENTO ---
        # Trasforma il cubo di dati 128x8x8 in un unico vettore piatto.
        # 128 * 8 * 8 = 8192 elementi.
        self.flatten = nn.Flatten()
        
        # --- POLICY HEAD (Il "Suggeritore di Mosse") ---
        # Decide quali sono le mosse migliori.
        self.policy_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512), # Strato denso (completamente connesso)
            nn.ReLU(),
            nn.Linear(512, 4096) # 4096 è un'approssimazione (64 caselle partenza * 64 arrivo)
            # Qui usciamo con dei punteggi (logit) per ogni possibile mossa.
        )
        
        # --- VALUE HEAD (Il "Giudice della Posizione") ---
        # Valuta se la posizione corrente è favorevole o meno.
        self.value_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1), # Un solo numero in uscita
            nn.Tanh() # Schiaccia il risultato tra -1 (perdi) e 1 (vinci)
        )

    def forward(self, x):
        """
        Definisce il percorso dei dati (Forward Pass):
        x -> Convoluzioni -> Appiattimento -> Policy & Value
        """
        x = self.conv_block(x)    # Estrae le caratteristiche spaziali
        x = self.flatten(x)       # Prepara i dati per i livelli densi
        
        policy = self.policy_head(x) # Calcola le probabilità delle mosse
        value = self.value_head(x)   # Calcola il valore della posizione
        
        return policy, value

# --- GESTIONE HARDWARE (RTX 4060) ---
# Controlla se la GPU è disponibile, altrimenti usa la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inizializza il modello e spostalo sulla memoria della GPU
model = ChessNet().to(device)

# Definiamo l'ottimizzatore: Adam è lo standard per iniziare
optimizer = optim.Adam(model.parameters(), lr=0.001)