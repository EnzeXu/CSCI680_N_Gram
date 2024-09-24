public class DangaMemcache implements Memcache { private static final Logger log = LoggerFactory.getLogger(DangaMemcache.class); private final MemCachedClient memcachedClient; private final String poolName; private MemcacheExceptionHandler exceptionHandler = new LoggingMemcacheExceptionHandler(); public DangaMemcache(MemCachedClient memcachedClient, String poolName) { this.memcachedClient = memcachedClient; this.poolName = poolName; } public Object get(String key) { try { log.debug("MemCachedClient.get({})", key); return memcachedClient.get(key); } catch (Exception e) { exceptionHandler.handleErrorOnGet(key, e); } return null; } public Map<String, Object> getMulti(String... keys) { try { return memcachedClient.getMulti(keys); } catch (Exception e) { exceptionHandler.handleErrorOnGet(StringUtils.join(keys, ", "), e); } return null; } public void set(String key, int cacheTimeSeconds, Object o) { log.debug("MemCachedClient.set({})", key); try { Calendar calendar = Calendar.getInstance(); calendar.setTime(new Date()); calendar.add(Calendar.SECOND, cacheTimeSeconds); memcachedClient.set(key, o, calendar.getTime()); } catch (Exception e) { exceptionHandler.handleErrorOnSet(key, cacheTimeSeconds, o, e); } } public void delete(String key) { try { memcachedClient.delete(key); } catch (Exception e) { exceptionHandler.handleErrorOnDelete(key, e); } } public void incr(String key, int factor, int startingValue) { try { long rv = memcachedClient.incr(key, factor); if (-1 == rv) memcachedClient.addOrIncr(key, startingValue); } catch (Exception e) { exceptionHandler.handleErrorOnIncr(key, factor, startingValue, e); } } public void shutdown() { log.debug("Shutting down danga MemCachedClient"); SockIOPool.getInstance(poolName).shutDown(); } public void setExceptionHandler(MemcacheExceptionHandler exceptionHandler) { this.exceptionHandler = exceptionHandler; } }