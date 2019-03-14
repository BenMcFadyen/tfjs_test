import { Component, OnInit } from '@angular/core';
import { Prediction } from './Prediction';

import * as tf from '@tensorflow/tfjs';
import {IMAGENET_CLASSES} from './ImageNetClasses';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent 
{
	title = 'tfjs_test';

	url: string;
	selectedModel: string;
	predictions: Prediction[];  // {Classname, Probability}

	InceptionV3: tf.Model;
	ResNet50: tf.Model;
	Xception : tf.Model;

	models = ["InceptionV3", "ResNet50", "Xception"];

	ngOnInit() 
	{
		this.loadModels()
	}

	async loadModels()
	{
		try 
		{
			this.InceptionV3 = await tf.loadModel('assets/KerasModels/InceptionV3/model.json');
			console.log('InceptionV3 Loaded');		

			this.ResNet50 = await tf.loadModel('assets/KerasModels/ResNet50/model.json');
			console.log('ResNet50 Loaded');		

			this.Xception = await tf.loadModel('assets/KerasModels/Xception/model.json');
			console.log('Xception Loaded');		
		}
		catch(e) 
		{
		  console.error("Error loading model: " + e);
		}			
	}

	onPredictButtonPress()
	{
		var selectedModelName = this.selectedModel;

		if(this.url == null)
		{
			console.error("No image selected");
			return;
		}

		switch (selectedModelName)
		{
			case "ResNet50":
			this.executeModel(this.ResNet50, 224);
				break;

			case "Xception":
			this.executeModel(this.Xception, 299);
				break;

			case "InceptionV3":
			this.executeModel(this.InceptionV3, 299);
				break;

			default:

				console.error("Cannot find model: " + selectedModelName)	
			break;
		}

	}

	async executeModel(model, imgSize)
	{
		var number_channels = 3;

		await tf.tidy(() => 
		{
			// Draw the image (with required dimensions) to the canvas
			this.drawIMGToCanvas(document.getElementById('img'), 'tensorCanvas', imgSize);

			var canvas = <HTMLCanvasElement> document.getElementById('tensorCanvas')

			let tensor = tf.browser.fromPixels(canvas, number_channels);
      	  	let normalizationOffset = tf.scalar(127.5);
            var normalized = tensor.toFloat().sub(normalizationOffset).div(normalizationOffset);
            var batched = normalized.reshape([1, imgSize, imgSize, 3]);

			//Make and format the predications
			var output = model.predict(batched) as any;
			console.log("Model Output: ")
			console.log(output)

			this.decodeModelOutput(output, 10);
				
		});
	}


	drawIMGToCanvas(img, canvasID, imgSize)
	{
		var canvas = <HTMLCanvasElement> document.getElementById(canvasID);
		canvas.height = imgSize;
		canvas.width = imgSize;		
		var context = canvas.getContext("2d");
		context.drawImage(img, 0, 0, imgSize, imgSize);
	}


	decodeModelOutput(output, topX)
	{
		var modelOutput = Array.from(output.dataSync()); 

		console.log("Model Predictions:")
		console.log(modelOutput)

		var preds: Prediction[];
		preds = new Array();

		for(var i = 0; i < modelOutput.length; i++)
		{
			preds[i] = (new Prediction(IMAGENET_CLASSES[i], modelOutput[i]))
		}

		// Sort predictions, DSC by probability
		preds = preds.sort(function(a,b){return a.probability < b.probability?1:a.probability >b.probability?-1:0})

		this.predictions = new Array();
		for(var i = 0; i < topX; i++)
		{
			this.predictions[i] = preds[i];
		}

		console.log('Top ' + topX + ' predictions: ');
		console.log(this.predictions);			
	}	

 	onSelectFile(event) 
	{ 	
	    const file = event.target.files[0]
		// called each time file input changes
		if (event.target.files && event.target.files[0]) 
		{
   			const file = event.target.files[0]

			var reader = new FileReader();
			reader.readAsDataURL(event.target.files[0]); // read file as data url

			// called once readAsDataURL is completed
			reader.onload = (event:any) => {this.url = event.target.result;} //sets <img url to new file url
		}
	}
}
