"""Production CPG Euler flux selection shared by MMS matrix recipes."""
FLUXES = ('lfr', 'Roe', 'HLL', 'Chandrashekar', 'Ranocha', 'Pirozzoli')


def select_fluxes(selection):
    if selection.strip().lower() == 'all':
        return FLUXES
    aliases = {name.lower(): name for name in FLUXES}
    aliases['llf'] = 'lfr'
    names = [name.strip().lower() for name in selection.split(',')]
    if not names or any(name not in aliases for name in names):
        raise ValueError('Choose all or a comma-separated subset of: ' + ', '.join(FLUXES))
    result = tuple(aliases[name] for name in names)
    if len(set(result)) != len(result):
        raise ValueError('Duplicate flux selection')
    return result
