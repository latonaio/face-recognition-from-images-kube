#!/usr/bin/env python3
# coding: utf-8

# Copyright (c) Latona. All rights reserved.

import json
import os
import sys

# Azure Face API用モジュール
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

# AION共通モジュール
from StatusJsonPythonModule import StatusJsonRest

BASE_PATH = os.path.join(os.path.dirname(__file__), )
PERSON_GROUP_ID = "latona-vr"

class FaceRecognition():
    def __init__(self):
        settings = json.load(
            open(os.path.join(os.path.dirname(__file__), 'face-api-config.json'), 'r')
        )
        # Create an authenticated FaceClient.
        self.face_client = FaceClient(
            settings.get('API_ENDPOINT'), 
            CognitiveServicesCredentials(settings.get('API_ACCESS_KEY'))
        )

    def getPersonIDFromImage(self, faceImage):
        image = open(faceImage, 'r+b')

        # Detect faces
        face_ids = []
        faces = self.face_client.face.detect_with_stream(image)
        for face in faces:
            face_ids.append(face.face_id)
        if not face_ids:
            return []

        person_list = []
        persons = self.face_client.face.identify(face_ids, PERSON_GROUP_ID)
        for person in persons:
            if person.candidates:
                candidate = person.candidates[0] # itiban match takai
                for face in faces:
                    if person.face_id == face.face_id:
                        person_list.append({
                            'person_id': candidate.person_id,
                            'confidence': candidate.confidence,
                            'face_rectangle': {
                                'width': face.face_rectangle.width,
                                'height': face.face_rectangle.height,
                                'left': face.face_rectangle.left,
                                'top': face.face_rectangle.top,
                            },
                        })

        return person_list

def main():
    # Jsonファイルをロードする
    statusObj = StatusJsonRest.StatusJsonRest(os.getcwd(), __file__)
    statusObj.initializeInputStatusJson()

    # 看板からVR画像パスを取得
    vr_image = statusObj.getMetadataFromJson("filepath")
    #vr_image = './test/face_test_image01.jpg'

    # FaceAPIに接続
    fr = FaceRecognition()
    # Face:どの人物に該当するかを判定する
    person_list = fr.getPersonIDFromImage(vr_image)

    statusObj.initializeOutputStatusJson()
    statusObj.setNextService(
        "GetInformationFromFaceMaster",
        "/home/latona/luna/Runtime/get-information-from-face-master",
        "python", "main.py")
    statusObj.setMetadataValue("image", vr_image)
    statusObj.setMetadataValue("person_list", person_list)
    statusObj.outputJsonFile()


if __name__ == "__main__":
    main()
