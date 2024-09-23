public class MemcachedCache implements Cache { private final Logger log = LoggerFactory.getLogger(MemcachedCache.class); private final String regionName; private final Memcache memcache; private final String clearIndexKey; private int cacheTimeSeconds = 300; private boolean clearSupported = false; private KeyStrategy keyStrategy = new Sha1KeyStrategy(); private boolean dogpilePreventionEnabled = false; private double dogpilePreventionExpirationFactor = 2; public static final Integer DOGPILE_TOKEN = 0; public MemcachedCache(String regionName, Memcache memcachedClient) { this.regionName = (regionName != null) ? regionName : "default"; this.memcache = memcachedClient; clearIndexKey = this.regionName.replaceAll("\\s", "") + ":index_key"; } public int getCacheTimeSeconds() { return cacheTimeSeconds; } public void setCacheTimeSeconds(int cacheTimeSeconds) { this.cacheTimeSeconds = cacheTimeSeconds; } public boolean isClearSupported() { return clearSupported; } public void setClearSupported(boolean clearSupported) { this.clearSupported = clearSupported; } public boolean isDogpilePreventionEnabled() { return dogpilePreventionEnabled; } public void setDogpilePreventionEnabled(boolean dogpilePreventionEnabled) { this.dogpilePreventionEnabled = dogpilePreventionEnabled; } public double getDogpilePreventionExpirationFactor() { return dogpilePreventionExpirationFactor; } public void setDogpilePreventionExpirationFactor(double dogpilePreventionExpirationFactor) { if (dogpilePreventionExpirationFactor < 1.0) { throw new IllegalArgumentException("dogpilePreventionExpirationFactor must be greater than 1.0"); } this.dogpilePreventionExpirationFactor = dogpilePreventionExpirationFactor; } private String dogpileTokenKey(String objectKey) { return objectKey + ".dogpileTokenKey"; } private Object memcacheGet(Object key) { String objectKey = toKey(key); if (dogpilePreventionEnabled) { return getUsingDogpilePrevention(objectKey); } else { log.debug("Memcache.get({})", objectKey); return memcache.get(objectKey); } } private Object getUsingDogpilePrevention(String objectKey) { Map<String, Object> multi; String dogpileKey = dogpileTokenKey(objectKey); log.debug("Checking dogpile key: [{}]", dogpileKey); log.debug("Memcache.getMulti({}, {})", objectKey, dogpileKey); multi = memcache.getMulti(dogpileKey, objectKey); if ((multi == null) || (multi.get(dogpileKey) == null)) { log.debug("Dogpile key ({}) not found updating token and returning null", dogpileKey); memcache.set(dogpileKey, cacheTimeSeconds, DOGPILE_TOKEN); return null; } return multi.get(objectKey); } private void memcacheSet(Object key, Object o) { String objectKey = toKey(key); int cacheTime = cacheTimeSeconds; if (dogpilePreventionEnabled) { String dogpileKey = dogpileTokenKey(objectKey); log.debug("Dogpile prevention enabled, setting token and adjusting object cache time. Key: [{}]", dogpileKey); memcache.set(dogpileKey, cacheTimeSeconds, DOGPILE_TOKEN); cacheTime = (int) (cacheTimeSeconds * dogpilePreventionExpirationFactor); } log.debug("Memcache.set({})", objectKey); memcache.set(objectKey, cacheTime, o); } private String toKey(Object key) { return keyStrategy.toKey(regionName, getClearIndex(), key); } public Object read(Object key) throws CacheException { return memcacheGet(key); } public Object get(Object key) throws CacheException { return memcacheGet(key); } public void put(Object key, Object value) throws CacheException { memcacheSet(key, value); } public void update(Object key, Object value) throws CacheException { put(key, value); } public void remove(Object key) throws CacheException { memcache.delete(toKey(key)); } public void clear() throws CacheException { if (clearSupported) { memcache.incr(clearIndexKey, 1, 1); } } public void destroy() throws CacheException { } public void lock(Object key) throws CacheException { } public void unlock(Object key) throws CacheException { } public long nextTimestamp() { return System.currentTimeMillis() / 100; } public int getTimeout() { return cacheTimeSeconds; } public String getRegionName() { return regionName; } public long getSizeInMemory() { return -1; } public long getElementCountInMemory() { return -1; } public long getElementCountOnDisk() { return -1; } public Map<?,?> toMap() { throw new UnsupportedOperationException(); } public String toString() { return "Memcached (" + regionName + ")"; } private long getClearIndex() { Long index = null; if (clearSupported) { Object value = memcache.get(clearIndexKey); if (value != null) { if (value instanceof String) { index = Long.valueOf((String) value); } else if (value instanceof Long) { index = (Long) value; } else { throw new IllegalArgumentException( "Unsupported type [" + value.getClass() + "] found for clear index at cache key [" + clearIndexKey + "]"); } } if (index != null) { return index; } } return 0L; } public KeyStrategy getKeyStrategy() { return keyStrategy; } public void setKeyStrategy(KeyStrategy keyStrategy) { this.keyStrategy = keyStrategy; } }