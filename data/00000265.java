public class StringKeyStrategy extends AbstractKeyStrategy { protected String transformKeyObject(Object key) { String stringKey = String.valueOf(key); log.debug("Transformed key [{}] to string [{}]", key, stringKey); return stringKey; } }