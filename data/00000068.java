public class BaseCacheMap<V> implements Map<String, V> { private final String keyPrefix; private final Transcoder<V> transcoder; private final MemcachedClientIF client; private final int exp; public BaseCacheMap(MemcachedClientIF c, int expiration, String prefix, Transcoder<V> t) { super(); keyPrefix = prefix; transcoder = t; client = c; exp = expiration; } public void clear() { throw new UnsupportedOperationException(); } private String getKey(String k) { return keyPrefix + k; } public boolean containsKey(Object key) { return get(key) != null; } public boolean containsValue(Object value) { return false; } public Set<Map.Entry<String, V>> entrySet() { return Collections.emptySet(); } public V get(Object key) { V rv = null; try { rv = client.get(getKey((String) key), transcoder); } catch (ClassCastException e) { } return rv; } public boolean isEmpty() { return false; } public Set<String> keySet() { return Collections.emptySet(); } public void putAll(Map<? extends String, ? extends V> t) { for (Map.Entry<? extends String, ? extends V> me : t.entrySet()) { client.set(getKey(me.getKey()), exp, me.getValue()); } } public V remove(Object key) { V rv = null; try { rv = get(key); client.delete(getKey((String) key)); } catch (ClassCastException e) { } return rv; } public int size() { return 0; } public Collection<V> values() { return Collections.emptySet(); } public V put(String key, V value) { V rv = get(key); client.set(getKey(key), exp, value); return rv; } }