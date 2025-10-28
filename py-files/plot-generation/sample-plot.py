#!/usr/bin/env python

import pandas as pd 
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import subprocess

# hyper paremers for plotting
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 
######
# olak = r"WI"

path="/data/yizhangh/ldp-btf/sigmod-submission"

naive__ = r"Baseline1"
naive_label="^"

bs__ = r"Baseline2"
bs__label="o"

adv_1 = r"Single-Source"
adv__label_1="s"

adv_2 = r"Double-Source"
adv__label_2="s"

adv_3 = r"Double-Source-Imbalanced"
adv__label_3="s"

LINE_='-'
color_q0 = '#cc3333'
color_bs = '#e95814'
color_q1 = 'black'
color_q2 = 'black'
color_q3 = 'black'
color_q4 = '#236133'
font1 = {'size' : 20}    

def get_pos_and_labels(indices):
    positions = []
    labels = []
    for i in indices:
        positions.append(pow(10, i))
        label = f"$10^{{{'-' if i < 0 else ''}{abs(i)}}}$"
        labels.append(label)
    return tuple(positions), tuple(labels)
dir=""

## base functions 
def plot_line(a, data,COLOR,LINESTYLE,MAKER,LABEL):
    plt.plot(a,data,color=COLOR,linestyle=LINESTYLE, markersize=15, 
             markeredgewidth =1.5,markerfacecolor='none',marker=MAKER,label = LABEL)
    
def plot_line_(data,COLOR,LINESTYLE,MAKER,LABEL):
    a = [i*0.2 for i in range(1,6)]
    plt.plot(a,data,color=COLOR,linestyle=LINESTYLE, markersize=10, markerfacecolor='none',marker=MAKER,label = LABEL)

def plt_settings():
    plt.style.use('default')
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams["legend.framealpha"] = 0
    plt.rcParams["legend.handletextpad"] = 0.1
    plt.rcParams["legend.columnspacing"] = 0.2
    # for the varying plots 
    plt.rcParams["figure.figsize"] = (6,5)
    plt.rcParams['pdf.fonttype'] = 42

# fig 7(a)
def plot__(a, data,COLOR,LINESTYLE,MAKER,LABEL):
    plt.plot(a,data,color=COLOR,linestyle=LINESTYLE, markersize=15, 
            markeredgewidth =1.5,markerfacecolor='none',marker=MAKER,label = LABEL)

# filename = "default_queries.pdf"
def saveas(filename):
    plt.tight_layout()
    plt.savefig("/data/yizhangh/ldp-btf/sigmod-submission/plots/"+filename)
    plt.close()

def default_plot():        
    # dataset_list =['dbpedia-location', 'edit-dewiki', 'dblp-author', 'lrcwiki', 'discogs_lstyle_lstyle', 
    #             'dbpedia-producer', 'live', 'dbpedia-team', 'orkut-groupmemberships', 
    #             'stackexchange-stackoverflow', 'delicious-ui', 'dbpedia-occupation', 
    #             'bookcrossing_full-rating', 'dbpedia-writer', 'opsahl-collaboration', 
    #             'bag-kos', 'epinions-rating', 'netflix', 'edit-stwiktionary', 
    #             'librec-filmtrust-ratings', 'unicode', 'nips', 
    #             'dbpedia-starring', 'discogs', 'rmwiki', 
    #             'movielens-10m_rating', 'github', 'wiki-en-cat', 'amazon-ratings', 
    #             'edit-nowiktionary', 'bpywiki', 'edit-iowiktionary', 
    #             'digg-votes', 'tewiktionary', 'discogs_genre_genre', 
    #             'lastfm_band', 'csbwiki'
    # ]
    dataset_list= [
        # 'lrcwiki', 
        # 'librec-filmtrust-ratings', 
        'rmwiki', 
        'opsahl-collaboration', 
        # 'csbwiki', 
        # 'dbpedia-producer', 
        'dbpedia-occupation', 
        'dbpedia-location', 
        'bag-kos',
        'bpywiki', 
        # 'github', 
        'tewiktionary', 
        # 'edit-nowiktionary', 
        # 'lastfm_band', 
        'bookcrossing_full-rating', 
        'stackexchange-stackoverflow', 
        'dbpedia-team', 
        "digg-votes",
        'wiki-en-cat', 
        # 'amazon-ratings', 
        # 'dblp-author', 
        'movielens-10m_rating', 'epinions-rating', 
        # 'edit-dewiki', 
        'netflix', 'delicious-ui', 
        # 'live', 
        'orkut-groupmemberships'
    ]

    # Initialize an empty list to store dictionaries of data
    data_list = []

    for data in dataset_list:
        filename = data + "-compare-new.txt"
        # Execute shell command to grep 'mae' from file
        grep_process = subprocess.Popen(['cat', filename], stdout=subprocess.PIPE)
        output = subprocess.check_output(['grep', 'mae'], stdin=grep_process.stdout, text=True)
        grep_process.wait()
        
        # Extract numbers
        numbers = [float(num) for num in output.split() if num.replace('.', '').isdigit()]
        # Create a dictionary for the dataset
        data_dict = {
            "Data": data,
            "naive": numbers[0],
            "baseline": numbers[1],
            "adv1": numbers[2],
            "adv2": numbers[5]
        }
        
        # Append the dictionary to the list
        data_list.append(data_dict)

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)

    # Set the 'Data' column as index
    df.set_index('Data', inplace=True)

    # Add additional columns from meta files
    for data in dataset_list:
        # Read the meta file
        with open(f"../../bidata/{data}.meta", "r") as meta_file:
            lines = meta_file.readlines()
            
            n1, n2, m = map(int, lines)
            # Add columns to DataFrame
            df.loc[data, 'n1'] = n1
            df.loc[data, 'n2'] = n2
            df.loc[data, 'm'] = m
            df.loc[data, 'fillrate'] = m/(n1*n2)

    # Sort the DataFrame by "m" column in ascending order
    # df = df.sort_values(by='m', ascending=True)
    # Create the new columns
    df['r1'] = df['naive'] / df['baseline']
    df['r2'] = df['baseline'] / df['adv1']
    df['r3'] = df['adv1'] / df['adv2']

    # df = df.sort_values(by=["r2", "r1", "r3"], ascending=False)
    df = df.sort_values(by=["m"], ascending=True)

    print("len = ", len(df))

    # file_path = 'output.xlsx'
    # # Save the DataFrame to Excel
    # df.to_excel(file_path, index=True)

    print(df.index.tolist())

    # Print or use the DataFrame as needed
    # print(df)
    print(df.iloc[:, 0:])  # Assuming you want to exclude the first three columns

    ## take the best 15 datasets, sort by |E|
    # df = df.tail(15)
    # df = df.sort_values(by='m', ascending=True)

    LIMIT = 10 * pow(10, 5)
    plt.style.use('default')
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams["legend.framealpha"] = 0
    plt.rcParams["figure.figsize"] = (10, 3.5)
    plt.rcParams["legend.handletextpad"] = 0.1
    total_width, n = 0.9, 4
    width = total_width / (n + 1)

    # Extract the datasets and MAE values from the DataFrame
    name_list = df.index.tolist()
    name_list = [name[0].upper() + name[-1].upper() for name in df.index]

    x1 = df['naive'].tolist()
    x2 = df['baseline'].tolist()
    x3 = df['adv1'].tolist()
    x4 = df['adv2'].tolist()

    # Calculate the positions for the bars
    x = [i for i in range(len(name_list))]

    # Plot the bars for each dataset
    for i in range(len(x)):
        # Adjust x positions for each set of bars
        x[i] = x[i] + width

        # Plot bars
        plt.bar(x[i], x1[i], width=width, linewidth=0.5, edgecolor='black', fc='white')
        plt.bar(x[i] + width, x2[i], width=width, linewidth=0.5, hatch="", edgecolor='black', fc='lightgrey')
        plt.bar(x[i] + 2*width, x3[i], width=width,  linewidth=0.5, hatch="////", edgecolor='black', fc='grey')
        plt.bar(x[i] + 3*width, x4[i], width=width, linewidth=0.5, hatch="////"*2, edgecolor='black', fc='black')

    # Create a single legend
    # plt.legend(fontsize=16, ncol=4, loc="upper right")
    # Add legend
    plt.legend([naive__, bs__, adv_1, adv_2], fontsize=16, ncol=2, loc='upper left')


    plt.yscale('log')

    plt.xticks([i+2.5*width for i in range(len(name_list))], name_list, fontsize=18)

    indices = [i for i in range(0, 8, 2)]
    pos, labels = get_pos_and_labels(indices)
    plt.yticks(pos, labels, fontsize=20)

    plt.xlabel("datasets", fontsize=20)
    plt.ylabel("mean absolute error", fontsize=20)
    plt.rcParams["legend.columnspacing"] = 0.3
    # plt.legend(fontsize=16, ncol=5, loc="upper right")
    plt.tight_layout()

    ## download this picture
    plt.savefig("plots/"+"default-plot.pdf")
    plt.close()

def vary_plot(data):
    plt_settings()

    print("processing ", data)
    # Define the shell command for bs1
    bs1_command = "cat "+data+"-vary.txt | grep mae | grep naive | awk '{print $4}' | uniq"
    # Execute the shell command for bs1
    bs1_result = subprocess.run(bs1_command, shell=True, capture_output=True, text=True)
    bs1 = [float(element) for element in bs1_result.stdout.splitlines()]

    # Define the shell command for bs2
    bs2_command = "cat "+data+"-vary.txt | grep mae | grep bs | awk '{print $4}' | uniq"
    # Execute the shell command for bs2
    bs2_result = subprocess.run(bs2_command, shell=True, capture_output=True, text=True)
    bs2 = [float(element) for element in bs2_result.stdout.splitlines()]

    # Define the shell command for adv
    adv_command = "cat "+data+"-vary.txt | grep mae | grep adv | awk '{print $4}'"
    # Execute the shell command for adv
    adv_result = subprocess.run(adv_command, shell=True, capture_output=True, text=True)
    adv__ = [float(element) for element in adv_result.stdout.splitlines()]

    # Separate adv into adv1 and adv2
    adv1 = adv__[::2] 
    adv2 = adv__[1::2]

    # Create a DataFrame
    df = pd.DataFrame({
        'bs1': bs1, 
        'bs2': bs2,
        'adv1': adv1,
        'adv2': adv2
    })
    
    a = [1, 1.5, 2, 2.5, 3]  # Assuming the index contains the epsilon values

    x1 = df['bs1'].tolist()[-5:]  # Extract mean relative errors for bs1
    x2 = df['bs2'].tolist()[-5:]  # Extract mean relative errors for bs2
    x3 = df['adv1'].tolist()[-5:] # Extract mean relative errors for adv1
    x4 = df['adv2'].tolist()[-5:]  # Extract mean relative errors for adv2

    plot_line(a, x1, color_q3, LINE_, naive_label, naive__)
    plot_line(a, x2, color_q3, LINE_, bs__label, bs__)
    # plot_line(a, x3, color_q3, LINE_, adv__label_1, adv_1)
    plot_line(a, x4, color_q3, LINE_, adv__label_2, adv_2)  


    # Combine all mean relative error lists into one list
    all_errors = x1 + x2 + x3 + x4
    # Calculate the y-axis range based on the maximum and minimum values of all_errors
    y_min = np.min(all_errors)
    y_max = np.max(all_errors)
    # Adjust y-axis range for better visualization
    y_range = y_max - y_min
    y_min_adjusted = y_min/10
    y_max_adjusted = y_max*10

    plt.xticks(a, fontsize=20)
    plt.ylim(y_min_adjusted, y_max_adjusted)  # Adjusted y-axis range
    plt.yscale('log')

    indices = [ i for i in range(math.floor(math.log10(y_min_adjusted)), math.ceil(math.log10(y_max_adjusted)) + 1, 2)]
    pos, labels = get_pos_and_labels(indices)
    plt.yticks(pos, labels, fontsize=20)

    plt.rcParams["legend.columnspacing"] = 0.3
    plt.subplots_adjust(top=0.9, left=0.2, bottom=0.13)  # Adjust the left parameter as needed

    # plt.tight_layout()

    plt.legend(fontsize=15, ncol=1, loc="upper right")
    plt.ylabel("mean absolute error", fontsize=20)
    plt.xlabel(r"$\epsilon$", fontsize=20)
    plt.savefig("plots/"+data+"-vary.pdf")
    plt.close()


def vary_kappa_plot(data):
    plt_settings()
    print("processing ", data)
    file_path = "/data/yizhangh/ldp2/sigmod-submission/" + data+"-balance3.txt"
    # Construct the command for adv1
    adv1_command = f"cat {file_path} | grep adv1 | awk '{{print $4}}'"
    # Execute the shell command for adv1
    adv1_result = subprocess.run(adv1_command, shell=True, capture_output=True, text=True)
    adv1 = [float(element) for element in adv1_result.stdout.splitlines()]
    # print("adv1:", adv1)

    # Construct the command for adv2
    adv2_command = f"cat {file_path} | grep adv2 | awk '{{print $4}}'"
    # Execute the shell command for adv2
    adv2_result = subprocess.run(adv2_command, shell=True, capture_output=True, text=True)
    adv2 = [float(element) for element in adv2_result.stdout.splitlines()]
    # print("adv2:", adv2)

    # Construct the command for adv3
    adv3_command = f"cat {file_path} | grep adv3 | awk '{{print $4}}'"
    # Execute the shell command for adv3
    adv3_result = subprocess.run(adv3_command, shell=True, capture_output=True, text=True)
    adv3 = [float(element) for element in adv3_result.stdout.splitlines()]
    # print("adv3:", adv3)

    print("ratio = ", adv2[-1]/adv3[-1])

    total_width, n = 0.9, 3
    width = total_width / (n + 1)
    # name_list = [1, 1.5, 2, 2.5, 3]
    x = [i for i in range(4)]
    # name_list = ["10^{}".format(i) for i in x]
    name_list = ["$10^{}$".format(i) for i in x]

    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, adv1, width=width, label='adv1', linewidth=0.5, edgecolor='black', fc='white')

    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, adv2, width=width, label='adv2', linewidth=0.5, hatch="", edgecolor='black', fc='lightgrey')

    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, adv3, width=width, label='adv3', linewidth=0.5, hatch="////"*2, edgecolor='black', fc='grey')

    # Combine all mean relative error lists into one list
    all_errors = adv1 + adv2 + adv3
    # Calculate the y-axis range based on the maximum and minimum values of all_errors
    y_min = np.min(all_errors)
    y_max = np.max(all_errors)
    # Adjust y-axis range for better visualization
    y_range = y_max - y_min
    y_min_adjusted = y_min/10
    y_max_adjusted = y_max*10

    plt.ylim(y_min_adjusted, y_max_adjusted)
    plt.yscale('log')

    indices = [ i for i in range(math.floor(math.log10(y_min_adjusted)), math.ceil(math.log10(y_max_adjusted)) + 1, 2)]
    pos, labels = get_pos_and_labels(indices)
    plt.yticks(pos, labels, fontsize=20)

    plt.xticks([i+1.5*width for i in range(len(name_list))], name_list, fontsize=18)

    plt.xlabel(r"$\kappa$", fontsize=20)
    plt.ylabel("mean relative error", fontsize=20)
    plt.rcParams["legend.columnspacing"] = 0.3
    plt.legend(fontsize=15, ncol=5, loc="upper right")
    plt.tight_layout()
    plt.savefig("balance-plots/"+ data + "-adv-comparison.pdf")
    plt.close()


def find_plot(data):
    plt_settings()

    print("processing ", data)

    file_path = "/data/yizhangh/ldp-btf/sigmod-submission/" + data+"-find2.txt"
    ## next time we will read from find2
    bs1_command = f"cat {file_path} | awk '{{print $5}}'"

    bs1_result = subprocess.run(bs1_command, shell=True, capture_output=True, text=True)
    bs1 = [float(element) for element in bs1_result.stdout.splitlines()]

    print(bs1)

    a = [i/10 for i in range(1, 9)]

    adv_mae_values = bs1[:8]
    smart_adv_mae = bs1[-1]

    plot_line(a, adv_mae_values, color_q3, LINE_, naive_label, "Double-source-basic")

    plt.axhline(y=smart_adv_mae, color='r', linestyle='--', label = "Double-source")

    # Add labels and title
    plt.xticks(a[::2], fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r"$\epsilon_1$", fontsize=20)
    plt.ylabel("mean absolute error", fontsize=20)
    # plt.title('Adv MAE vs Epsilon for lrcwiki dataset')
    plt.legend(fontsize=15, ncol=1, loc="upper right")
    plt.rcParams["legend.columnspacing"] = 0.3
    plt.subplots_adjust(top=0.9, left=0.2, bottom=0.15) 
    plt.savefig("find-plots/"+ data + "-find.pdf")
    plt.close()


vary_list = ["unicode", "rmwiki", "dbpedia-location", "dbpedia-team", 
    "opsahl-collaboration", "lrcwiki", "librec-filmtrust-ratings", "csbwiki", "bag-kos", 
    "bpywiki", "nips", "lastfm_band", "discogs_lstyle_lstyle", "digg-votes", 
    "movielens-10m_rating", "netflix", "amazon-ratings", "bookcrossing_full-rating", 
    "dblp-author", "dbpedia-occupation", "dbpedia-producer", "dbpedia-starring", 
    "dbpedia-writer", "delicious-ui",
     "edit-dewiki",
    "edit-iowiktionary",
    # "edit-nowiktionary",
    "edit-stwiktionary",
    "epinions-rating",
    "github",
    "lastfm_band",
    "live",
    "lrcwiki",
    "orkut-groupmemberships",
    "stackexchange-stackoverflow",
    "tewiktionary",
    "wiki-en-cat"
    ]

# evaluate the effect of epsilon
# for data in vary_list:
#     vary_plot(data)



balance_list = [    "bag-kos",
    "bpywiki",
    "nips",
    "lastfm_band",
    # "discogs_lstyle_lstyle",
    "digg-votes",
    "stackexchange-stackoverflow",
    "tewiktionary",
    "discogs",
    "edit-dewiki",
    # "edit-iowiktionary"
    # "edit-nowiktionary"
    # "edit-stwiktionary"
    "epinions-rating",
    "github",
    ]

for data in balance_list:
    vary_kappa_plot(data)

default_plot()

find_list = ["unicode", "rmwiki", "dbpedia-location", "dbpedia-team", "opsahl-collaboration", 
    "lrcwiki", "librec-filmtrust-ratings", "csbwiki", "bag-kos", "bpywiki", "nips", 
    "lastfm_band", "discogs_lstyle_lstyle", "digg-votes", "movielens-10m_rating", 
    "netflix", "amazon-ratings", "bookcrossing_full-rating", "dblp-author", "dbpedia-occupation", 
    "dbpedia-producer", "dbpedia-starring", "dbpedia-writer", "delicious-ui", "discogs_genre_genre", 
    "discogs_lstyle_lstyle", "discogs", "edit-dewiki", "edit-iowiktionary", "edit-nowiktionary", 
    "edit-stwiktionary", "epinions-rating", "github", "lastfm_band",
    "live","lrcwiki","orkut-groupmemberships","stackexchange-stackoverflow","tewiktionary","wiki-en-cat"    
]

for data in find_list[:24]:
    find_plot(data)

# i think we need not only apply smart strategy on dense graph. we should do it for all.

