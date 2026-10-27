/** Convert the x-major occupancy grid into canvas coordinates (y points upward in the map). */
export function mapPixels(cells) {
  const height = cells[0]?.length || 0;
  return cells.flatMap((column, x) => column.map((value, y) => ({x, y: height - y - 1, value})));
}

/** World (x, y) metres to canvas pixel coordinates for a snapshot map; null when outside it. */
export function worldToPixel(map, x, y) {
  const width = map.cells.length;
  const height = map.cells[0]?.length || 0;
  const px = (x - map.origin_m[0]) / map.resolution_m;
  const py = height - (y - map.origin_m[1]) / map.resolution_m;
  return px < 0 || py < 0 || px > width || py > height ? null : {px, py};
}
