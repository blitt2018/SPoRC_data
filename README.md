Welcome to the github repository for the creation of SPoRC: the Structured Podcast Research Corpus!

You can access our data [here](https://huggingface.co/datasets/blitt/SPoRC), our data processing pipeline [here](https://github.com/blitt2018/SPoRC_data), and our publication [here](FILL_IN).

To create our dataset, we begin with podcast-level metadata from [Podcast Index](https://podcastindex.org/) and collect episode-level metadata by scraping the RSS feeds associated with each English podcast from May-June 2020. We then feed mp3 url's from these RSS feeds into a three-phase pipeline that extracts transcript, audio, and speaker-turn information. Finally, all of these data types are merged together at the episode level and speaker-turn level and [released](https://huggingface.co/datasets/blitt/SPoRC) for future non-commercial use.

## Our 3-Phase Pipeline 
Here, we release code our three-phase pipeline such that transcripts, audio features, and speaker turns can be given a particular mp3 url as input. 

## Transcription 
