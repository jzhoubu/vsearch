from wordcloud import WordCloud
import matplotlib.pyplot as plt

def wordcloud_from_dict(d: dict, max_words=100, width=800, height=800, save_file=None):
    wordcloud = WordCloud(max_words=max_words, width=width, height=height).generate_from_frequencies(d)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    if save_file is not None:
        wordcloud.to_file(save_file)

