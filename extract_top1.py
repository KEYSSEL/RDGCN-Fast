import os
def extract_top1(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"❌ Erreur : Le fichier d'entrée '{input_file}' n'existe pas.")
        return

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # 1. Lire et écrire l'en-tête
        header = f_in.readline()
        f_out.write(header)
        
        # 2. Filtrer les lignes
        count = 0
        for line in f_in:
            parts = line.strip().split('\t')
            # Le rang est dans la 3ème colonne (index 2)
            if len(parts) >= 3 and parts[2] == '1':
                f_out.write(line)   
                count += 1
                
    print(f"✅ Extraction terminée ! {count} alignements Top-1 ont été sauvegardés dans '{output_file}'.")

base_dir = "EAkit/results"
liste_fichiers = []

for dossier in os.listdir(base_dir):
    chemin = os.path.join(base_dir, dossier, "final_alignments_top10.txt")
    
    if os.path.isfile(chemin):
        liste_fichiers.append(chemin)
    else:
        print(f"❌ Fichier manquant : {chemin}")

for fichier in liste_fichiers:
    output_file = fichier.replace("top10", "top1")
    extract_top1(fichier, output_file)
