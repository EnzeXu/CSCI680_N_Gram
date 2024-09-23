public class GeoHashDecodingBenchmark { private static final int NUMBER_OF_HASHES = 1000000; private static final char[] base32 = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' }; private String[] randomHashes; @Before public void setup(){ randomHashes = new String[NUMBER_OF_HASHES]; Random rand = new Random(); for(int i = 0; i < NUMBER_OF_HASHES; i++){ int characters = rand.nextInt(10) + 2; StringBuilder string = new StringBuilder(); for(int j = 0; j < characters; j++){ string.append(base32[rand.nextInt(base32.length)]); } randomHashes[i] = string.toString(); } } @Benchmark public void benchmarkRandomDecoding(){ for(String hash : randomHashes){ GeoHash geoHash = GeoHash.fromGeohashString(hash); } } }