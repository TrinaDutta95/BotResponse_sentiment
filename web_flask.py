import bot_response as br
from flask import Flask, request
from flask import jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def generate_response():
    text = request.args.get("user_res")
    print(text)
    bot_res = br.sentence_processing(text)
    print(bot_res)
    return jsonify(bot_res)


if __name__ == "__main__":
    app.run()
