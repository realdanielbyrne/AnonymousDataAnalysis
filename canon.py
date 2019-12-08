# canon - Classify Anonymous Data
import logging
import azure.functions as func

def main(req: func.HttpRequest) -> func.HttpResponse:
  req_body = req.get_json()
