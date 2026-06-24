import itertools
def build_triplets(source_df, labels, n=N):
    triplets_list = []
    for label1, label2 in itertools.combinations(labels, 2):

        df_class1 = source_df[source_df['LCSH_Label'] == label1]
        df_class2 = source_df[source_df['LCSH_Label'] == label2]

        if len(df_class1) == 0 or len(df_class2) == 0:
            continue

        anchor_samples = df_class1.sample(n=n, replace=True).reset_index(drop=True)
        positive_samples = df_class1.sample(n=n, replace=True).reset_index(drop=True)
        negative_samples = df_class2.sample(n=n, replace=True).reset_index(drop=True)

        for i in range(n):
            triplets_list.append({
                'input_anchor': anchor_samples.iloc[i]['tfidf_vector'],
                'input_positive': positive_samples.iloc[i]['tfidf_vector'],
                'input_negative': negative_samples.iloc[i]['tfidf_vector']
            })
    return triplets_list
