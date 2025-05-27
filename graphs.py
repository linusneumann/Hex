import matplotlib.pyplot as plt
import json
import numpy as np

with open("plotting/history_4_sparse_cce_64_loss1.json", "r") as f:
    hist1 = json.load(f)

with open("plotting/11x11x5_4_d5_w128_sparse_cce_dense_64_acc1.json", "r") as f:
    hist2 = json.load(f)
with open("plotting/11x11x3_4_d5_w128_sparse_cce_dense_64_acc1.json", "r") as f:
    hist3 = json.load(f)
# Plot Loss
with open("top_k_pred.txt","r") as f:
    hist4 = f.readline().split(';')

acc = []
topk= []
for pred in hist4:
   
    accv, k = pred.split(',')
    acc.append(accv)
    topk.append(int(k))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
#plt.scatter(topk,acc, label='Validation Accuracy')

plt.plot(hist1['loss'], label='Train Loss')
plt.plot(hist1['val_loss'], label='Validation Loss')
#plt.plot(hist2['val_loss'], label='d=5, 11x11x5 Validation Loss')
#plt.plot(hist3['loss'], label='d=6 Train Loss')
#plt.plot(hist3['val_loss'], label='d=5, 11x11x3 Validation Loss')
plt.xlabel('Epoche',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Verlustverlauf (Loss)')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(hist1['accuracy'], label='Train Accuracy')
plt.plot(hist1['val_accuracy'], label='Validation Accuracy')
#plt.plot(hist2['accuracy'], label='d=5 Train Accuracy')
#plt.plot(hist2['val_accuracy'], label='d=5, 11x11x5 Validation Accuracy')
#plt.plot(hist3['accuracy'], label='d=6 Train Accuracy')
#plt.plot(hist3['val_accuracy'], label='d=5, 11x11x3 Validation Accuracy')
plt.xlabel('Epoche',fontsize=16)
plt.ylabel('Genauigkeit',fontsize=16)
plt.title('Genauigkeitsverlauf (Accuracy)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

if __name__ == "__main__":
    
    with open("tournament/rndabsplay-4_2_7.json", "r") as f:
        data = json.load(f)
        player1_mt = []
        player2_mt = []
        player1_mt_sideswitch = []
        player2_mt_sideswitch = []
        player1_red_wins = 0
        player2_red_wins = 0
        games = data[0]["games"]
        len1 = len(games)
        for game in games:
            player1_mt.append(game["mean_times"]["player1"])
            player2_mt.append(game["mean_times"]["player2"])
            if game["win"] == "red":
                player1_red_wins = player1_red_wins+1
        print(len1)
        games = data[1]["games"]
        len2 = len(games)
        for game in games:
            player1_mt_sideswitch.append(game["mean_times"]["player1"])
            player2_mt_sideswitch.append(game["mean_times"]["player2"])
            if game["win"] == "red":
                player2_red_wins = player2_red_wins+1
        print(len2)
        print(f"player1 red winrate:{player1_red_wins/len1}")
        print(f"player2 red winrate:{player2_red_wins/len2}")

    with open("tournament/rndabsplay-4_2_7.json", "r") as f:
        data = json.load(f)
        player1_mtnoord = []
        player2_mtnoord = []
        player1_mt_sideswitchnoord = []
        player2_mt_sideswitchnoord = []
        player1_red_wins = 0
        player2_red_wins = 0
        games = data[0]["games"]
        len1 = len(games)
        for game in games:
            player1_mtnoord.append(game["mean_times"]["player1"])
            player2_mtnoord.append(game["mean_times"]["player2"])
            if game["win"] == "red":
                player1_red_wins = player1_red_wins+1
        print(len1)
        games = data[1]["games"]
        len2 = len(games)
        for game in games:
            player1_mt_sideswitchnoord.append(game["mean_times"]["player1"])
            player2_mt_sideswitchnoord.append(game["mean_times"]["player2"])
            if game["win"] == "red":
                player2_red_wins = player2_red_wins+1
        print(len2)
        print(f"player1 red winrate:{player1_red_wins/len1}")
        print(f"player2 red winrate:{player2_red_wins/len2}")

    # Boxplot zur Veranschaulichung der Verteilung der mittleren Zeiten
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(player1_mt,  label="Player 1, d=4")
    plt.plot(player2_mt, label="Player 2, d=2")
    #plt.plot(player1_mtnoord,  label="Player 1, d=2, no ordering")
    #plt.plot(player2_mtnoord, label="Player 2, d=2")
    plt.title("Verteilung der mittleren Zugzeiten")
    plt.ylabel("Zeit pro Zug (Sekunden)",fontsize=16)
    plt.xlabel("Spiele",fontsize=16)
    plt.grid(True)
    plt.legend()
    #plt.xlim([0,19])
    plt.xticks(np.arange(0,len1,1))
    plt.title('Vor Seitenwechsel')

    plt.subplot(1,2,2)
    plt.plot(player2_mt_sideswitch, label="Player 2, d=4")
    plt.plot(player1_mt_sideswitch,  label="Player 1, d=2")
    
    #plt.plot(player1_mt_sideswitchnoord,  label="Player 1, d=2")
    #plt.plot(player2_mt_sideswitchnoord, label="Player 2, d=2, no ordering")
    plt.title("Verteilung der mittleren Zugzeiten")
    plt.ylabel("Zeit pro Zug (Sekunden)",fontsize=16)
    plt.xlabel("Spiele",fontsize=16)
    plt.grid(True)
    plt.legend()
    #plt.xlim([0,19])
    plt.xticks(np.arange(0,len2,1))
    plt.title('Nach Seitenwechsel')

    plt.tight_layout()
    plt.show()