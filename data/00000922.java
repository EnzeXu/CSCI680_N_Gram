public class BoundingBoxGeoHashIteratorTest { @Test public void testIter() { BoundingBox box = new BoundingBox(37.7, 37.84, -122.52, -122.35); BoundingBoxGeoHashIterator iter = new BoundingBoxGeoHashIterator( TwoGeoHashBoundingBox.withBitPrecision(box, 10)); checkIterator(iter); } @Test public void testIter2() { BoundingBox box = new BoundingBox(37.7, 37.84, -122.52, -122.35); BoundingBoxGeoHashIterator iter = new BoundingBoxGeoHashIterator( TwoGeoHashBoundingBox.withBitPrecision(box, 35)); checkIterator(iter); } @Test public void testIter3() { BoundingBox box = new BoundingBox(28.5, 67.15, -33.2, 44.5); BoundingBoxGeoHashIterator iter = new BoundingBoxGeoHashIterator( TwoGeoHashBoundingBox.withCharacterPrecision(box, 2)); List<GeoHash> hashes = checkIterator(iter); assertThat(hashes.size(), is(49)); } @Test public void testEndlessIterator() { BoundingBox box = new BoundingBox(72.28907f, 88.62655f, -50.976562f, 170.50781f); TwoGeoHashBoundingBox twoGeoHashBoundingBox = TwoGeoHashBoundingBox.withCharacterPrecision(box, 2); BoundingBoxGeoHashIterator iterator = new BoundingBoxGeoHashIterator(twoGeoHashBoundingBox); Set<GeoHash> hashes = new HashSet<>(); while (iterator.hasNext()) { GeoHash hash = iterator.next(); assertThat("Hash has been already produced by the iterator once: " + hash, hashes, not(hasItem(hash))); hashes.add(hash); } } @Test public void testAllCells() { BoundingBox box = new BoundingBox(-90, 90, -180, 180); TwoGeoHashBoundingBox twoGeoHashBoundingBox = TwoGeoHashBoundingBox.withCharacterPrecision(box, 2); BoundingBoxGeoHashIterator iterator = new BoundingBoxGeoHashIterator(twoGeoHashBoundingBox); Set<GeoHash> hashes = new HashSet<>(); while (iterator.hasNext()) { GeoHash hash = iterator.next(); hashes.add(hash); } assertThat(hashes.size(), is(1024)); } @Test public void testTopRightCorner() { BoundingBox box = new BoundingBox(84.4, 84.9, 169.3, 179.6); TwoGeoHashBoundingBox twoGeoHashBoundingBox = TwoGeoHashBoundingBox.withCharacterPrecision(box, 2); BoundingBoxGeoHashIterator iterator = new BoundingBoxGeoHashIterator(twoGeoHashBoundingBox); Set<GeoHash> hashes = new HashSet<>(); while (iterator.hasNext()) { GeoHash hash = iterator.next(); assertThat("Hash has been already produced by the iterator once: " + hash, hashes, not(hasItem(hash))); hashes.add(hash); } } private List<GeoHash> checkIterator(BoundingBoxGeoHashIterator iter) { BoundingBox newBox = iter.getBoundingBox().getBoundingBox(); List<GeoHash> hashes = new ArrayList<>(); while (iter.hasNext()) { hashes.add(iter.next()); } GeoHash prev = null; for (GeoHash gh : hashes) { if (prev != null) { Assert.assertTrue(prev.compareTo(gh) < 0); } Assert.assertTrue(newBox.contains(gh.getOriginatingPoint())); prev = gh; } return hashes; } }