# Neo4j Setup Guide for Slovak Health Knowledge Graph

## Quick Setup Options

### Option 1: Neo4j Desktop (Recommended for beginners)
1. Download Neo4j Desktop from: https://neo4j.com/download/
2. Install and create a new database
3. Set password (remember this!)
4. Start the database
5. Open Neo4j Browser

### Option 2: Docker (Quick start)
```bash
# Run Neo4j in Docker
docker run \
    --name neo4j-health \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/healthgraph123 \
    neo4j:5.23
```

### Option 3: Neo4j Aura (Cloud)
1. Go to: https://console.neo4j.io/
2. Create free AuraDB instance
3. Note connection details

## Connection Details

Once Neo4j is running, note these details:
- **URI**: `bolt://localhost:7687` (or your cloud URI)
- **Username**: `neo4j`
- **Password**: Your chosen password
- **Browser URL**: `http://localhost:7474`

## Next Steps

After setup, our Python script will:
1. Connect to your Neo4j database
2. Create the Slovak health knowledge graph
3. Provide you with visualization URL

## Troubleshooting

### Common Issues:
- **Port conflicts**: Change ports if 7474/7687 are in use
- **Memory**: Increase Docker memory if needed
- **Authentication**: Ensure username/password are correct

### Quick Test:
Open Neo4j Browser and run:
```cypher
CREATE (test:Test {name: "Hello World"})
RETURN test
```

If this works, you're ready to proceed!