#http://www.astro.caltech.edu/~mcs/CBI/pointing/
import numpy as np
import matplotlib

import datetime


def local_coords_dhl(dec_r, ha_r, lat_r):

    sE = np.sin(dec_r) * np.sin(lat_r) + np.cos(dec_r) * np.cos(lat_r) * np.cos(ha_r)
    E = np.arcsin(sE)

    tA1, tA2 = -np.sin(ha_r) * np.cos(dec_r),  (np.sin(dec_r) * np.cos(lat_r) - np.cos(dec_r) * np.sin(lat_r) * np.cos(ha_r))
    A = np.arctan2(tA1, tA2)

    tP1, tP2 = np.sin(ha_r) * np.cos(lat_r), (np.sin(lat_r) * np.cos(dec_r) - np.sin(dec_r) * np.cos(lat_r) * np.cos(ha_r))
    P = np.arctan2(tP1, tP2)

    return E, A, P



def hms_to_rad(h, m, s):
    return (h + m / 60.0 + s / 3600.0) / 24.0 * 2 * np.pi

def dms_to_rad(d, m, s):
    return (d + m / 60.0 + s / 3600.0) / 180.0 * np.pi


def julian_day(year, month, day):

    jd = 367 * year - np.trunc(7 * (year + np.trunc((month+9.0)/12.0))/4.0) + np.trunc((275.0*month)/9.0) + day + 1721013.5 - 0.5*np.sign(100*year+month-190002.5) + 0.5# + UT/24 

    return jd


def gst(year, month, day, time):
    t = 6.656306 + 0.0657098242*(julian_day(year, month, day)-2445700.5) + 1.0027379093*time

    return t


def lst(year, month, day, time, longitude):
    t = gst(year, month, day, time)

    l = (t + longitude * (24.0 / 360.0)) % 24.0

    return l


def lst_gmrt(year, month, day, time):

    ut = time - 5.5

    return lst(year, month, day, ut, 74.0497)

def lst_today(h, m):
    return lst_gmrt(2013, 1, 16.0, h + m / 60.0) / 24.0 * 2 *  np.pi