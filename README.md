*This is a submission for the [Open Source AI Challenge with pgai and Ollama ](https://dev.to/challenges/pgai)*

## What I Built

I picked an application which would resonate to dev community, which demonstrate how ai applications are built using embeddings, vector databases and also provides a robust starting point for ai projects using docker and docker compose file including timescaledb and ollama:

* timescaledb
* pgvector and pgai
* ollama container
* fastapi provides the apis and serves static part

I wanted to explore all the options so the application works with the ollama or openai, via pgai extension, and it can also work with a cloud database from timescale, with pgai vectorizer.

The online version is hosted on a simple vps instance so it's configured to work with openai api, due to resource constraints.

## Demo

I decided to publish the prototype as a site:

https://gitcone.com/


## Tools Used

The website is using python to segment all the files from a repository into smaller chunks and they are stored in the database instance. It can use either the local timescaledb database with the pgai(pgvector) included, to store the chunks with their respective embeddings. 

The application makes use of the pgai functions to either use ollama or open openai, making switching between configurations, very easy.

In an early version the application was using pgai vectorizer in the timescale cloud database, but because it was working only with openai, I used standard pgai approach with separate embeddings table.

## Final Thoughts

I was pleasant surprised of how easy was to develop this prototype and about the fact it worked smoothly. Dockerizing the all of them was challenging but it brings a lot of benefits.

Prize Categories:
Open-source Models from Ollama, Vectorizer Vibe, All the Extensions

## Further developments

For better selecting the context, I'm looking into adding additional queries and additional fields in the data chunks, that would work well extracting relevant data based on the user intent. 

Timescale with pgai seems to fit very well multi agents architectures, as those api calls can go well as database queries without additional configurations.

Thanks for the challenge!