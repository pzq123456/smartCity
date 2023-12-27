var elevation = SRTM30.select('elevation').clip(SD);
var slope = ee.Terrain.slope(elevation);
 // 可视化参数
 var args = {
   crs: 'EPSG:3857',
   dimensions: '300',
   region: SD,
   min: -2000,
   max: 10000,
   palette: 'green, blanchedalmond,orange,black ',
   framesPerSecond: 12,
 };
//  Map.addLayer(elevation,args,'elevation');
//  Map.addLayer(slope,{},'slope');

Export.image.toDrive({
  image: elevation,   
  scale: 30, 
  maxPixels: 1e13,
  region: SD });

Export.image.toDrive({
  image: slope,
  scale: 30,
  maxPixels: 1e13,
  region: SD });