//VERSION=3
// Adaptacion cuantitativa del script oficial "Cyanobacteria Chlorophyll-a NDCI".
// Devuelve valores numericos, no una imagen RGB, para permitir el analisis temporal.
// Fuente: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/cyanobacteria_chla_ndci_l1c/

function setup() {
  return {
    input: [{ bands: ["B03", "B04", "B05", "B08", "SCL", "dataMask"], units: "REFLECTANCE" }],
    output: [
      { id: "indices", bands: 4, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(s) {
  const badScl = [0, 1, 3, 8, 9, 10, 11].includes(s.SCL);
  const valid = s.dataMask === 1 && !badScl;
  const ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
  const ndwi = (s.B03 - s.B08) / (s.B03 + s.B08);
  const ndci = (s.B05 - s.B04) / (s.B05 + s.B04);

  // Ecuacion implementada en el evalscript oficial de CyanoLakes.
  const rawChla = 826.57 * Math.pow(ndci, 3) - 176.43 * Math.pow(ndci, 2) + 19 * ndci + 4.071;
  const cyanoChla = Math.max(0, Math.min(500, rawChla));
  const water = s.SCL === 6 || ndwi > 0.05;
  const keep = valid && water;

  return {
    indices: keep ? [cyanoChla, ndci, ndvi, ndwi] : [NaN, NaN, NaN, NaN],
    dataMask: [keep ? 1 : 0]
  };
}

