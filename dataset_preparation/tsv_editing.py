import pandas as pd


for part in range(100):
    # load tab separated csv
    df = pd.read_csv(r'/xml/survey_vol4.csv', sep='\t', header=0, engine="python",
                     encoding="UTF8")

    start_index = df[df['name'] == 'Tekst 1'].index[0]
    end_index = df.index[-1]

    slice = df[start_index:end_index + 1].copy()
    part += 1
    picked_texts = pd.read_csv(rf'data\picked_all_{part}.csv')
    texts = picked_texts['text'].tolist()

    len(set(texts))

    for idx, text in enumerate(texts):
        to_modify = slice.copy()
        replacement_text = text
        section_name = f'Header {idx + 1}'
        q_1_name = f'Q_1_{idx + 1}'
        q_2_name = f'Q_2_{idx + 1}'
        q_3_name = f'Q_3_{idx + 1}'
        q_4_name = f'Q_4_{idx + 1}'

        to_replace = 'Kiedy patrzę na zdjęcia tych ludzi nie czuję absolutnie żadnych emocji.'
        to_modify.loc[to_modify.index == 78, 'text'] = to_modify[to_modify.index == 78]['text'].iloc[0].replace(to_replace, replacement_text)
        to_modify.loc[to_modify.index == 90, 'text'] = to_modify[to_modify.index == 90]['text'].iloc[0].replace(to_replace, replacement_text)
        to_modify.loc[to_modify.index == 104, 'text'] = to_modify[to_modify.index == 104]['text'].iloc[0].replace(to_replace, replacement_text)

        to_modify.loc[to_modify['name'] == 'Tekst 1', 'type/scale'] = idx + 1
        to_modify.loc[to_modify['name'] == 'Tekst 1', 'name'] = section_name

        to_modify.loc[to_modify['name'] == 'Q1', 'name'] = q_1_name
        to_modify.loc[to_modify['name'] == 'Q2', 'name'] = q_2_name
        to_modify.loc[to_modify['name'] == 'Q3', 'name'] = q_3_name
        to_modify.loc[to_modify['name'] == 'Q4', 'name'] = q_4_name

        # now concatenate at the end of to_modify
        df = pd.concat([df[:start_index], to_modify], axis=0)
        start_index += to_modify.shape[0]

    columns = df.columns

    for col in columns:
        temp = []
        for i in df[col].values:
            try:
                temp.append(str(int(i)))
            except:
                temp.append(i)
        df[col] = temp

    # save
    df.to_csv(rf'D:\PycharmProjects\annotations\data\edited_survey_{part}.csv', sep='\t', index=False)





    # # load
    # df = pd.read_csv(r'D:\PycharmProjects\annotations\xml\edited_survey.csv', sep='\t', header=0, engine="python", encoding="UTF8")
    #
    #
    # columns = df.columns
    #
    # for col in columns:
    #     temp = []
    #     for i in df[col].values:
    #         try:
    #             temp.append(str(int(i)))
    #         except:
    #             temp.append(i)
    #     df[col] = temp
    # # save
    # df.to_csv(r'D:\PycharmProjects\annotations\xml\edited_survey_2.csv', sep='\t', index=False)