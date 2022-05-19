import json
import flask
from flask import jsonify, request
from flask_cors import CORS, cross_origin

from SearchEngine import SearchEngine


# Create Flask application method
app = flask.Flask(__name__)
cors = CORS(app)
# Start debugger
app.config["DEBUG"] = True

engine = SearchEngine()
engine.load_query_embedding()


@app.route('/', methods=["GET"])
def home():
    return "<h1>Embeddings search Engine</h1><p>Dual Space Word Embeddings Semantic Search Engine</p>"


@app.route('/status', methods=["GET"])
@cross_origin()
def status():
    """Get search engine status"""
    return jsonify(status=str(engine.get_status()))


@app.route('/train', methods=["POST"])
@cross_origin()
def train():
    """Train semantic search model on API documentation"""
    data = request.get_data()
    entries = json.loads(data)['entries']
    engine.train(entries)
    response = jsonify(status=str(engine.get_status()))
    return response


@app.route('/search', methods=["GET"])
@cross_origin()
def search():
    """Search a query on trained model"""
    query_parameters = request.args
    query = query_parameters.get('query')
    results = engine.search(query, False)
    return json.dumps(results)


app.run()
